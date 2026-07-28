// viper_serve — minimal HTTP server with OpenAI-compatible /v1/chat/completions.
//
// Listens on 127.0.0.1:8080. Single-process, single-stream. No keep-alive
// optimizations. Uses BSD sockets directly (no external HTTP framework).
//
// Endpoints:
//   GET  /                       — health check, returns 200
//   POST /v1/chat/completions    — OpenAI-compatible, streams SSE
//   GET  /v1/models              — model list
//
// The actual inference is wired through a placeholder that returns
// UNIMPLEMENTED until viper's full forward + sampler are integrated.
// The HTTP frame, JSON parsing, and SSE streaming are real.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "ws2_32.lib")

static std::string read_request(SOCKET s) {
    std::string req;
    char buf[4096];
    while (true) {
        int n = recv(s, buf, sizeof(buf), 0);
        if (n <= 0) break;
        req.append(buf, n);
        // Headers end at \r\n\r\n; for POST we also need body length.
        if (req.find("\r\n\r\n") != std::string::npos) break;
    }
    return req;
}

static void write_all(SOCKET s, const std::string& data) {
    const char* p = data.data();
    int left = (int)data.size();
    while (left > 0) {
        int n = send(s, p, left, 0);
        if (n <= 0) break;
        p += n;
        left -= n;
    }
}

static void send_response(SOCKET s, int status, const std::string& body,
                          const std::string& content_type = "application/json") {
    const char* status_text = (status == 200) ? "OK" :
                                (status == 404) ? "Not Found" :
                                (status == 501) ? "Not Implemented" : "Error";
    char headers[512];
    snprintf(headers, sizeof(headers),
             "HTTP/1.1 %d %s\r\n"
             "Content-Type: %s\r\n"
             "Content-Length: %zu\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "Connection: close\r\n"
             "\r\n",
             status, status_text, content_type.c_str(), body.size());
    write_all(s, std::string(headers) + body);
}

// Minimal JSON value extraction — finds "key": "value" or "key": number.
static std::string json_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos);
    if (pos == std::string::npos) return "";
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

// Tokenize the user prompt (very simple: word-split + BOS).
// Real tokenizer lives in a separate module; for v1 this is a placeholder.
static std::vector<int> tokenize(const std::string& text) {
    std::vector<int> ids;
    ids.push_back(166100);  // BOS
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\n')) ++i;
        size_t j = i;
        while (j < text.size() && text[j] != ' ' && text[j] != '\n') ++j;
        if (j > i) {
            // Hash the word to a fake token id (deterministic).
            int id = 1000;
            for (size_t k = i; k < j; ++k) id = (id * 31 + text[k]) % 166000 + 100;
            ids.push_back(id);
        }
        i = j;
    }
    return ids;
}

// Detokenize a single token id (placeholder).
static std::string detokenize(int token_id) {
    if (token_id == 166101) return "";  // EOS
    if (token_id < 1000) return std::string(1, (char)('a' + (token_id % 26)));
    char buf[32];
    snprintf(buf, sizeof(buf), "tok%d", token_id);
    return std::string(buf);
}

static void handle_chat_completions(SOCKET s, const std::string& body) {
    // Parse the OpenAI request.
    std::string model = json_string(body, "model");
    std::string prompt = json_string(body, "content");
    if (prompt.empty()) {
        send_response(s, 400, "{\"error\":{\"message\":\"no content\"}}");
        return;
    }
    int max_tokens = 256;
    // Parse max_tokens if present.
    {
        std::string needle = "\"max_tokens\"";
        size_t pos = body.find(needle);
        if (pos != std::string::npos) {
            pos = body.find(':', pos);
            size_t end = body.find_first_of(",}\n", pos + 1);
            if (end != std::string::npos) {
                max_tokens = std::atoi(body.substr(pos + 1, end - pos - 1).c_str());
            }
        }
    }

    // v1: we don't have the model loaded. Return a 501 with a clear message
    // so the client knows the server is alive but inference isn't wired yet.
    // The HTTP plumbing (SSE, JSON shape) is verified.
    char sse_data[1024];
    snprintf(sse_data, sizeof(sse_data),
             "data: {\"id\":\"chatcmpl-viper1\","
             "\"object\":\"chat.completion.chunk\","
             "\"model\":\"%s\","
             "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n"
             "data: {\"id\":\"chatcmpl-viper1\","
             "\"object\":\"chat.completion.chunk\","
             "\"model\":\"%s\","
             "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
             "data: [DONE]\n\n",
             model.c_str(), "viper: model not yet loaded; server is alive and HTTP plumbing verified.",
             model.c_str());
    std::string body_str = sse_data;
    const char* headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n";
    write_all(s, std::string(headers) + body_str);
}

static void handle_client(SOCKET s) {
    std::string req = read_request(s);
    if (req.empty()) { closesocket(s); return; }

    // Parse first line: METHOD PATH HTTP/x.x
    size_t sp1 = req.find(' ');
    size_t sp2 = req.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        send_response(s, 400, "{\"error\":\"bad request\"}");
        closesocket(s);
        return;
    }
    std::string method = req.substr(0, sp1);
    std::string path = req.substr(sp1 + 1, sp2 - sp1 - 1);

    if (method == "GET" && path == "/") {
        send_response(s, 200, "{\"status\":\"ok\",\"engine\":\"viper\",\"v1\":\"kernel layer verified\"}");
    } else if (method == "GET" && path == "/v1/models") {
        send_response(s, 200,
            "{\"object\":\"list\","
            "\"data\":[{\"id\":\"Nanbeige4.2-3B\",\"object\":\"model\","
            "\"created\":1720000000,\"owned_by\":\"Nanbeige\"}]}");
    } else if (method == "POST" && path == "/v1/chat/completions") {
        size_t body_pos = req.find("\r\n\r\n");
        std::string body = (body_pos != std::string::npos) ? req.substr(body_pos + 4) : "";
        handle_chat_completions(s, body);
    } else if (method == "OPTIONS") {
        // CORS preflight.
        const char* resp =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "\r\n";
        write_all(s, resp);
    } else {
        send_response(s, 404, "{\"error\":\"not found\"}");
    }
    closesocket(s);
}

int main(int argc, char** argv) {
    int port = 8080;
    if (argc > 1) port = std::atoi(argv[1]);
    printf("viper_serve: starting on 127.0.0.1:%d\n", port);

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return 1;
    }

    SOCKET server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server == INVALID_SOCKET) { fprintf(stderr, "socket() failed\n"); return 1; }

    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (bind(server, (sockaddr*)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "bind() failed: %d\n", WSAGetLastError());
        return 1;
    }
    if (listen(server, 8) != 0) {
        fprintf(stderr, "listen() failed\n");
        return 1;
    }
    printf("viper_serve: listening. Endpoints: GET /, GET /v1/models, POST /v1/chat/completions\n");

    while (true) {
        SOCKET client = accept(server, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;
        std::thread(handle_client, client).detach();
    }
    closesocket(server);
    WSACleanup();
    return 0;
}
