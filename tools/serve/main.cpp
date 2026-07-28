// viper_serve — HTTP server with the REAL engine integrated.
// Loads .viper at startup, streams real tokens via SSE.
//
// Build: tools\serve\build_serve_cuda.bat
// Run:   viper_serve.exe --model artifacts\Nanbeige4.2-3B.viper --vocab artifacts\vocab.bin

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#include "viper/model_impl.cuh"
#include "viper/tokenizer.h"

static viper::NanbeigeEngine* g_engine = nullptr;
static viper::Tokenizer* g_tok = nullptr;

static std::string read_request(SOCKET s) {
    std::string req;
    char buf[4096];
    while (true) {
        int n = recv(s, buf, sizeof(buf), 0);
        if (n <= 0) break;
        req.append(buf, n);
        if (req.find("\r\n\r\n") != std::string::npos) {
            size_t cl_pos = req.find("Content-Length:");
            if (cl_pos != std::string::npos) {
                int cl = std::atoi(req.c_str() + cl_pos + 15);
                size_t body_start = req.find("\r\n\r\n") + 4;
                while ((int)req.size() < (int)body_start + cl) {
                    n = recv(s, buf, sizeof(buf), 0);
                    if (n <= 0) break;
                    req.append(buf, n);
                }
            }
            break;
        }
    }
    return req;
}

static void write_all(SOCKET s, const std::string& data) {
    const char* p = data.data();
    int left = (int)data.size();
    while (left > 0) {
        int n = send(s, p, left, 0);
        if (n <= 0) break;
        p += n; left -= n;
    }
}

static void send_response(SOCKET s, int status, const std::string& body,
                          const char* ct = "application/json") {
    const char* st = (status == 200) ? "OK" : (status == 404) ? "Not Found" : "Error";
    char h[512];
    snprintf(h, sizeof(h), "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
             "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
             status, st, ct, body.size());
    write_all(s, std::string(h) + body);
}

static std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s(n, '\0'); fread(s.data(), 1, n, f); fclose(f);
    return s;
}

static std::string json_string(const std::string& j, const std::string& key) {
    std::string n = "\"" + key + "\"";
    size_t pos = j.find(n);
    if (pos == std::string::npos) return "";
    pos = j.find('"', pos + n.size());
    if (pos == std::string::npos) return "";
    size_t end = j.find('"', pos + 1);
    return (end == std::string::npos) ? "" : j.substr(pos + 1, end - pos - 1);
}

static int json_int(const std::string& j, const std::string& key, int dflt) {
    std::string n = "\"" + key + "\"";
    size_t pos = j.find(n);
    if (pos == std::string::npos) return dflt;
    pos = j.find(':', pos + n.size());
    return (pos == std::string::npos) ? dflt : std::atoi(j.c_str() + pos + 1);
}

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

static void handle_chat(SOCKET s, const std::string& body) {
    std::string user_msg = json_string(body, "content");
    int max_tokens = json_int(body, "max_tokens", 128);
    if (max_tokens <= 0 || max_tokens > 2048) max_tokens = 128;

    std::string full = "<|im_start|>user\n" + user_msg + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int32_t> ids = g_tok->encode(full);

    write_all(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                 "Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n"
                 "Connection: close\r\n\r\n");

    // Prefill
    int32_t next = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
        bool last = (i + 1 == ids.size());
        if (!g_engine->forward(ids[i], last, &next)) {
            write_all(s, "data: {\"error\":\"forward failed\"}\n\n");
            closesocket(s); return;
        }
    }

    // Generate and stream
    int n_gen = 0;
    auto tg0 = std::chrono::steady_clock::now();
    for (int i = 0; i < max_tokens; ++i) {
        if (next == g_tok->eos() || next == g_tok->im_end()) break;
        std::string piece = g_tok->decode(next);
        if (!piece.empty()) {
            char sse[2048];
            snprintf(sse, sizeof(sse),
                "data: {\"id\":\"viper\",\"object\":\"chat.completion.chunk\","
                "\"model\":\"Nanbeige4.2-3B\",\"choices\":[{\"index\":0,"
                "\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                json_escape(piece).c_str());
            write_all(s, sse);
        }
        ++n_gen;
        if (!g_engine->forward(next, true, &next)) break;
    }
    auto tg1 = std::chrono::steady_clock::now();
    double gen_s = std::chrono::duration<double>(tg1 - tg0).count();

    char done[512];
    snprintf(done, sizeof(done),
        "data: {\"id\":\"viper\",\"object\":\"chat.completion.chunk\","
        "\"model\":\"Nanbeige4.2-3B\",\"choices\":[{\"index\":0,"
        "\"delta\":{},\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%d,"
        "\"timing\":{\"gen_s\":%.2f,\"tps\":%.1f}}}\n\n"
        "data: [DONE]\n\n",
        ids.size(), n_gen, gen_s, gen_s > 0 ? n_gen / gen_s : 0.0);
    write_all(s, done);
    closesocket(s);
    g_engine->reset();
}

static void handle_client(SOCKET s) {
    std::string req = read_request(s);
    if (req.empty()) { closesocket(s); return; }
    size_t sp1 = req.find(' ');
    size_t sp2 = req.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        send_response(s, 400, "{\"error\":\"bad request\"}");
        closesocket(s); return;
    }
    std::string method = req.substr(0, sp1);
    std::string path = req.substr(sp1 + 1, sp2 - sp1 - 1);

    if (method == "GET" && path == "/") {
        std::string html = read_file("D:/dev/viper/tools/serve/ui/chat.html");
        if (html.empty()) html = read_file("D:/dev/viper/tools/serve/ui.html");
        if (html.empty()) html = read_file("tools/serve/ui/chat.html");
        if (html.empty()) { send_response(s, 500, "{\"error\":\"ui\"}"); }
        else {
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                "Content-Length: " + std::to_string(html.size()) + "\r\nConnection: close\r\n\r\n";
            write_all(s, resp + html);
        }
    } else if (method == "GET" && path == "/health") {
        send_response(s, 200, "{\"status\":\"ok\",\"engine\":\"viper\"}");
    } else if (method == "GET" && path == "/v1/models") {
        send_response(s, 200, "{\"object\":\"list\",\"data\":[{\"id\":\"Nanbeige4.2-3B\"}]}");
    } else if (method == "POST" && path == "/v1/chat/completions") {
        size_t bp = req.find("\r\n\r\n");
        handle_chat(s, (bp != std::string::npos) ? req.substr(bp + 4) : "");
        return;
    } else if (method == "OPTIONS") {
        write_all(s, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                     "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                     "Access-Control-Allow-Headers: Content-Type\r\n\r\n");
    } else {
        send_response(s, 404, "{\"error\":\"not found\"}");
    }
    closesocket(s);
}

int main(int argc, char** argv) {
    std::string model_p = "D:/dev/viper/artifacts/Nanbeige4.2-3B.viper";
    std::string vocab_p = "D:/dev/viper/artifacts/vocab.bin";
    int port = 8080;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i+1 < argc) model_p = argv[++i];
        else if (std::strcmp(argv[i], "--vocab") == 0 && i+1 < argc) vocab_p = argv[++i];
        else if (std::strcmp(argv[i], "--port") == 0 && i+1 < argc) port = std::atoi(argv[++i]);
    }

    g_tok = new viper::Tokenizer();
    if (!g_tok->load(vocab_p)) { fprintf(stderr, "[serve] tok fail\n"); return 1; }
    g_engine = new viper::NanbeigeEngine();
    if (!g_engine->load(model_p)) { fprintf(stderr, "[serve] engine fail\n"); return 1; }

    printf("[serve] viper loaded. Port %d. Open http://127.0.0.1:%d/\n", port, port);

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { fprintf(stderr, "WSA fail\n"); return 1; }
    SOCKET server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server == INVALID_SOCKET) return 1;
    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (bind(server, (sockaddr*)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "bind fail: %d\n", WSAGetLastError()); return 1; }
    if (listen(server, 8) != 0) return 1;
    printf("[serve] listening on 127.0.0.1:%d\n", port);

    while (true) {
        SOCKET client = accept(server, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;
        std::thread(handle_client, client).detach();
    }
    closesocket(server);
    WSACleanup();
    return 0;
}
