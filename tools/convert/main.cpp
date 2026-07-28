// viper_convert — placeholder for M1. Prints usage.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "viper/common.h"
#include "viper/safety.h"

using namespace viper;

int main(int argc, char** argv) {
    if (argc == 1 || std::strcmp(argv[1], "--help") == 0 ||
        std::strcmp(argv[1], "-h") == 0) {
        std::puts("viper_convert — convert HF/GGUF weights to .viper format (M1)");
        std::puts("");
        std::puts("Usage: viper_convert --src <path> --dst <path> [--qtype Q4_G64|Q5_G64|Q6_G64|W8_G32|BF16]");
        std::puts("");
        std::puts("Flags:");
        std::puts("  --src <path>         Source model path (HF safetensors or GGUF)");
        std::puts("  --dst <path>         Destination .viper artifact");
        std::puts("  --qtype <type>       Target quantization (default BF16 — lossless)");
        std::puts("");
        std::puts("Skeleton phase: parses args, runs safety guard, exits with UNIMPLEMENTED.");
        return 0;
    }

    // Skeleton: parse a few flags, refuse if VRAM < 1 GiB, return 0 with a log.
    const char* src = nullptr;
    const char* dst = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--src") == 0 && i + 1 < argc) src = argv[++i];
        if (std::strcmp(argv[i], "--dst") == 0 && i + 1 < argc) dst = argv[++i];
    }
    if (!src || !dst) {
        std::fprintf(stderr, "missing --src or --dst\n");
        return 2;
    }

    safety::DeviceMem dm;
    if (auto s = safety::device_memory(dm); !s.ok()) {
        VIPER_LOG(Error, std::string("device_memory: ") + std::string(s.message()));
        return 3;
    }
    VIPER_LOG(Info, "convert " + std::string(src) + " -> " + std::string(dst));
    VIPER_LOG(Info, "free VRAM " + std::to_string(dm.free_bytes / (1 << 20)) + " MiB");
    VIPER_LOG(Info, "skeleton phase: real conversion lands in M1");
    return 0;
}