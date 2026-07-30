// viper_serve_cb — Continuous Batching HTTP Server
//
// Design:
//   Multiple NanbeigeEngine instances (one per concurrent sequence slot).
//   Each slot has its own KV cache and position tracking.
//   A worker thread pool processes requests from a queue.
//
// TRUE continuous batching (batching across sequences in one forward pass)
// requires per-sequence KV cache pointers in the batch attention kernel.
// This is a simplified version: parallel sequences, each with its own engine.
//
// For TRUE CB: modify forward_batch to accept per-token KV cache pointers
// and per-token position IDs. The batch attention kernel would index each
// token's KV cache independently. This is the architecture for v2.
//
// Build: Same as viper_serve but with this main file.
// Run:   viper_serve_cb.exe --model ... --vocab ... --slots 4
//
// API: OpenAI-compatible /v1/chat/completions with SSE streaming.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <atomic>

#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#include "viper/model_impl.cuh"
#include "viper/tokenizer.h"

// ─── Engine Pool ──────────────────────────────────────────────────────────

struct EngineSlot {
    viper::NanbeigeEngine engine;
    bool busy = false;
    SOCKET client_socket = INVALID_SOCKET;
    std::string prompt;
    int max_tokens = 256;
};

struct ServerState {
    std::vector<EngineSlot*> slots;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::queue<int> free_slots;  // indices into slots
    viper::Tokenizer* tok;
    int port = 9090;
};

static ServerState g_state;

void worker_thread(int slot_idx) {
    while (true) {
        int idx;
        {
            std::unique_lock<std::mutex> lock(g_state.queue_mutex);
            g_state.cv.wait(lock, [&] { return !g_state.free_slots.empty(); });
            idx = g_state.free_slots.front();
            g_state.free_slots.pop();
        }

        EngineSlot* slot = g_state.slots[idx];
        SOCKET client = slot->client_socket;

        // Tokenize prompt
        std::vector<int32_t> ids = g_state.tok->encode(slot->prompt);
        ids.insert(ids.begin(), g_state.tok->bos());

        // Reset engine for new sequence
        slot->engine.reset();

        // Prefill (sequential for correctness, batched for speed)
        int32_t next = -1;
        for (size_t i = 0; i < ids.size(); ++i) {
            bool last = (i + 1 == ids.size());
            if (!slot->engine.forward(ids[i], last, &next)) break;
        }

        // Stream tokens via SSE
        int n_gen = 0;
        auto t0 = std::chrono::steady_clock::now();
        while (n_gen < slot->max_tokens) {
            if (next == g_state.tok->eos() || next == g_state.tok->im_end()) break;

            std::string piece = g_state.tok->decode(next);
            // SSE format: data: {"choices":[{"delta":{"content":"..."}}]}
            std::string sse = "data: {\"choices\":[{\"delta\":{\"content\":\"";
            for (char c : piece) {
                if (c == '"') sse += "\\\"";
                else if (c == '\\') sse += "\\\\";
                else if (c == '\n') sse += "\\n";
                else sse += c;
            }
            sse += "\"}}]}\n\n";
            send(client, sse.c_str(), (int)sse.size(), 0);
            ++n_gen;

            if (!slot->engine.forward(next, true, &next)) break;
        }

        // End SSE stream
        const char* done = "data: [DONE]\n\n";
        send(client, done, (int)strlen(done), 0);

        // Close connection
        closesocket(client);
        slot->busy = false;

        // Return slot to pool
        {
            std::lock_guard<std::mutex> lock(g_state.queue_mutex);
            g_state.free_slots.push(idx);
        }
        g_state.cv.notify_one();

        auto t1 = std::chrono::steady_clock::now();
        double gen_s = std::chrono::duration<double>(t1 - t0).count();
        double tps = gen_s > 0 ? n_gen / gen_s : 0;
        printf("[slot %d] generated %d tokens at %.1f tok/s\n", idx, n_gen, tps);
    }
}

// ─── HTTP Handling ────────────────────────────────────────────────────────

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

static std::string extract_json_field(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";
    pos += pattern.size();
    size_t end = json.find("\"", pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
}

void handle_client(SOCKET client) {
    std::string req = read_request(client);

    // Check if it's a chat completion request
    if (req.find("/v1/chat/completions") != std::string::npos ||
        req.find("/v1/completions") != std::string::npos) {

        // Extract prompt from JSON body
        size_t body_start = req.find("\r\n\r\n");
        if (body_start == std::string::npos) {
            closesocket(client);
            return;
        }
        std::string body = req.substr(body_start + 4);
        std::string prompt = extract_json_field(body, "content");
        if (prompt.empty()) prompt = extract_json_field(body, "prompt");
        if (prompt.empty()) prompt = "Hello";

        // Find a free slot
        int slot_idx;
        {
            std::unique_lock<std::mutex> lock(g_state.queue_mutex);
            g_state.cv.wait(lock, [&] { return !g_state.free_slots.empty(); });
            slot_idx = g_state.free_slots.front();
            g_state.free_slots.pop();
        }

        // Assign request to slot
        g_state.slots[slot_idx]->client_socket = client;
        g_state.slots[slot_idx]->prompt = prompt;
        g_state.slots[slot_idx]->busy = true;

        // Send SSE headers
        const char* headers =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        send(client, headers, (int)strlen(headers), 0);

        // Wake up the worker for this slot
        {
            std::lock_guard<std::mutex> lock(g_state.queue_mutex);
            g_state.free_slots.push(slot_idx);
        }
        g_state.cv.notify_one();
    } else if (req.find("GET / ") != std::string::npos || req.find("GET /ui") != std::string::npos) {
        // Serve UI (redirect to existing chat.html)
        const char* resp = "HTTP/1.1 302 Found\r\nLocation: /chat.html\r\n\r\n";
        send(client, resp, (int)strlen(resp), 0);
        closesocket(client);
    } else {
        const char* resp = "HTTP/1.1 404 Not Found\r\n\r\n";
        send(client, resp, (int)strlen(resp), 0);
        closesocket(client);
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string modelp = "D:/dev/viper/artifacts/Nanbeige4.2-3B.viper";
    std::string vocabp = "D:/dev/viper/artifacts/vocab.bin";
    int n_slots = 2;  // number of concurrent sequence slots

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i+1 < argc) modelp = argv[++i];
        else if (std::strcmp(argv[i], "--vocab") == 0 && i+1 < argc) vocabp = argv[++i];
        else if (std::strcmp(argv[i], "--slots") == 0 && i+1 < argc) n_slots = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--port") == 0 && i+1 < argc) g_state.port = std::atoi(argv[++i]);
    }

    // Load tokenizer (shared)
    g_state.tok = new viper::Tokenizer();
    if (!g_state.tok->load(vocabp)) {
        fprintf(stderr, "[server] tokenizer load failed\n");
        return 1;
    }
    printf("[server] tokenizer loaded\n");

    // Create engine slots
    printf("[server] creating %d engine slots...\n", n_slots);
    for (int i = 0; i < n_slots; ++i) {
        EngineSlot* slot = new EngineSlot();
        slot->engine.max_batch = 8;
        if (!slot->engine.load(modelp)) {
            fprintf(stderr, "[server] engine %d load failed\n", i);
            return 1;
        }
        printf("[server] slot %d: engine loaded\n", i);
        g_state.slots.push_back(slot);
        g_state.free_slots.push(i);
    }

    // Start worker threads (one per slot)
    std::vector<std::thread> workers;
    for (int i = 0; i < n_slots; ++i)
        workers.emplace_back(worker_thread, i);

    // Initialize winsock
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    // Create listening socket
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(g_state.port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "[server] bind failed on port %d\n", g_state.port);
        return 1;
    }
    listen(server_fd, 16);

    printf("[server] listening on port %d with %d slots\n", g_state.port, n_slots);
    printf("[server] POST /v1/chat/completions with {\"content\":\"your prompt\"}\n");

    // Accept loop
    while (true) {
        SOCKET client = accept(server_fd, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;

        // Handle each connection in a separate thread
        std::thread(handle_client, client).detach();
    }

    // Cleanup (unreachable)
    closesocket(server_fd);
    WSACleanup();
    for (auto& w : workers) w.join();
    return 0;
}
