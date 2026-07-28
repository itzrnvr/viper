// viper_serve — minimal HTTP server stub. Binds to :8080, returns 501 for
// every request. Real routes (POST /v1/completions, GET /health) land in M5.
//
// Built without external deps — plain Winsock2 (since C++20 doesn't ship
// std::net yet). Link Ws2_32.

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#endif

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "viper/common.h"
#include "viper/safety.h"

using namespace viper;

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop = true; }

constexpr const char* k501 =
    "HTTP/1.1 501 Not Implemented\r\n"
    "Content-Type: text/plain\r\n"
    "Content-Length: 32\r\n"
    "Connection: close\r\n"
    "\r\n"
    "viper_serve: not implemented yet\r\n";

bool send_all(SOCKET s, const char* buf, int len) {
    while (len > 0) {
        int n = ::send(s, buf, len, 0);
        if (n <= 0) return false;
        buf += n;
        len -= n;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int port = 8080;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::puts("viper_serve — placeholder HTTP server (M5 lands routes)");
            std::puts("Usage: viper_serve [--port N]");
            return 0;
        }
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

#ifdef _WIN32
    WSADATA wsad;
    if (WSAStartup(MAKEWORD(2, 2), &wsad) != 0) {
        VIPER_LOG(Error, "WSAStartup failed");
        return 1;
    }
#endif

    SOCKET srv = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (srv == INVALID_SOCKET) {
        VIPER_LOG(Error, "socket() failed");
#ifdef _WIN32
        WSACleanup();
#endif
        return 2;
    }

    int yes = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes),
                 sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u16>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        VIPER_LOG(Error, "bind() failed on port " + std::to_string(port));
        ::closesocket(srv);
#ifdef _WIN32
        WSACleanup();
#endif
        return 3;
    }
    if (::listen(srv, 16) == SOCKET_ERROR) {
        VIPER_LOG(Error, "listen() failed");
        ::closesocket(srv);
#ifdef _WIN32
        WSACleanup();
#endif
        return 4;
    }

    VIPER_LOG(Info, "viper_serve listening on 127.0.0.1:" + std::to_string(port));
    VIPER_LOG(Info, "every route returns 501 — real handlers land in M5");

    while (!g_stop.load()) {
        sockaddr_in client{};
        int clen = sizeof(client);
        SOCKET cli = ::accept(srv, reinterpret_cast<sockaddr*>(&client), &clen);
        if (cli == INVALID_SOCKET) {
            if (g_stop.load()) break;
            continue;
        }
        char buf[2048];
        int n = ::recv(cli, buf, sizeof(buf), 0);
        (void)n;
        send_all(cli, k501, static_cast<int>(std::strlen(k501)));
        ::closesocket(cli);
    }

    ::closesocket(srv);
#ifdef _WIN32
    WSACleanup();
#endif
    VIPER_LOG(Info, "viper_serve exiting");
    return 0;
}