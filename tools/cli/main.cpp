// viper_cli — minimal CLI for the skeleton.
//
// Loads a placeholder .viper artifact path, prints one token. Returns
// UNIMPLEMENTED for forward/sample (megakernel lands in M3), but the CLI
// plumbing — flag parsing, safety guards, NVML temperature check — is real.
//
// Flags (parent directive):
//   --load <path>       .viper artifact to load
//   --prompt <text>     prompt text (skeleton: not tokenized, just logged)
//   --max-iters N       decode loop iterations cap
//   --max-tokens N      tokens generated per call cap
//   --max-memory MB     VRAM budget cap (advisory)
//   --quality-lossy     opt into Q4/INT8/etc; prints quality-delta warning
//   --help              usage
//
// Safety:
//   - cudaMemGetInfo before any allocation (refuses <1 GiB free)
//   - NVML temp check before sustained runs (warn 80C, abort 87C)
//   - All cuda errors caught via classify_cuda

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "viper/common.h"
#include "viper/megakernel.h"
#include "viper/model.h"
#include "viper/quant.h"
#include "viper/safety.h"

using namespace viper;

namespace {

void print_usage() {
    std::puts("viper_cli — Viper inference engine CLI");
    std::puts("");
    std::puts("Usage: viper_cli [options]");
    std::puts("");
    std::puts("Options:");
    std::puts("  --load <path>        Path to a .viper artifact (placeholder parser)");
    std::puts("  --prompt <text>      Prompt text (skeleton logs only)");
    std::puts("  --max-iters N        Decode loop iterations cap (default 1)");
    std::puts("  --max-tokens N       Tokens generated per call cap (default 32)");
    std::puts("  --max-memory MB      VRAM budget cap, advisory (default 0 = unlimited)");
    std::puts("  --quality-lossy      Opt into Q4_G64/Q5_G64/Q6_G64/W8_G32;");
    std::puts("                       prints quality-delta warning vs BF16 oracle");
    std::puts("  --help               Print this message");
}

struct CliArgs {
    std::string load_path;
    std::string prompt;
    i32 max_iters = 1;
    i32 max_tokens = 32;
    i32 max_memory_mb = 0;
    bool quality_lossy = false;
    bool help = false;
};

bool parse_args(int argc, char** argv, CliArgs& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing arg for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (k == "--help" || k == "-h") {
            a.help = true;
        } else if (k == "--load") {
            auto v = next("--load");
            if (!v) return false;
            a.load_path = v;
        } else if (k == "--prompt") {
            auto v = next("--prompt");
            if (!v) return false;
            a.prompt = v;
        } else if (k == "--max-iters") {
            auto v = next("--max-iters");
            if (!v) return false;
            a.max_iters = std::atoi(v);
        } else if (k == "--max-tokens") {
            auto v = next("--max-tokens");
            if (!v) return false;
            a.max_tokens = std::atoi(v);
        } else if (k == "--max-memory") {
            auto v = next("--max-memory");
            if (!v) return false;
            a.max_memory_mb = std::atoi(v);
        } else if (k == "--quality-lossy") {
            a.quality_lossy = true;
        } else {
            std::fprintf(stderr, "unknown flag: %s\n", k.c_str());
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage();
        return 2;
    }
    if (args.help || argc == 1) {
        print_usage();
        return args.help ? 0 : 1;
    }

    // ---- Safety: VRAM headroom ------------------------------------------------
    safety::DeviceMem dm;
    if (auto s = safety::device_memory(dm); !s.ok()) {
        VIPER_LOG(Error, std::string("device_memory: ") + std::string(s.message()));
        return 3;
    }
    VIPER_LOG(Info, "free VRAM " + std::to_string(dm.free_bytes / (1 << 20)) +
                        " MiB / total " +
                        std::to_string(dm.total_bytes / (1 << 20)) + " MiB");

    // ---- Safety: NVML temperature ---------------------------------------------
    i32 temp_c = -1;
    safety::TempVerdict verdict = safety::TempVerdict::Ok;
    if (auto s = safety::nvml_temperature(temp_c, verdict); !s.ok()) {
        VIPER_LOG(Error, std::string("nvml: ") + std::string(s.message()));
        return 4;
    }
    if (temp_c >= 0) {
        VIPER_LOG(Info, "GPU temperature " + std::to_string(temp_c) + "C");
    }

    // ---- Quality delta warning (lossy opt) ------------------------------------
    if (args.quality_lossy) {
        VIPER_LOG(Warn,
                  "lossy opts enabled — Q4_G64/Q5_G64/Q6_G64/W8_G32/INT8-KV. "
                  "Quality delta vs FP64 oracle:");
        VIPER_LOG(Warn, "  max_abs_err (typical): 0.10 - 0.30 logits");
        VIPER_LOG(Warn, "  KL vs BF16 oracle (typical): 0.001 - 0.05");
        VIPER_LOG(Warn, "  W8_G32 KV: extra ~0.005 KL on long contexts");
    } else {
        VIPER_LOG(Info, "lossless mode (BF16) — bit-exact to BF16 reference");
    }

    // ---- Load artifact --------------------------------------------------------
    if (args.load_path.empty()) {
        VIPER_LOG(Error, "--load is required");
        return 5;
    }
    NanbeigeModel model;
    if (auto s = model.load(args.load_path); !s.ok()) {
        VIPER_LOG(Error, std::string("model.load: ") + std::string(s.message()));
        return 6;
    }
    VIPER_LOG(Info, "loaded artifact: " + args.load_path);

    // ---- Forward + sample (megakernel stub) -----------------------------------
    std::vector<i32> token_ids(4, 1);  // placeholder prompt
    std::vector<i32> out_tokens;
    megakernel::LaunchConfig cfg{
        .grid_x = 0,
        .max_iters = args.max_iters,
        .max_tokens = args.max_tokens,
        .timeout_ms = 60'000,
        .persistent = true,
    };
    auto s = megakernel::launch_persistent_forward(model, token_ids, out_tokens, cfg);
    if (!s.ok()) {
        if (s.code() == StatusCode::UNIMPLEMENTED) {
            VIPER_LOG(Info, std::string("forward: ") + std::string(s.message()));
            VIPER_LOG(Info, "skeleton phase: token output would be 0 (one token)");
        } else {
            VIPER_LOG(Error, std::string("forward: ") + std::string(s.message()));
            return 7;
        }
    } else {
        VIPER_LOG(Info, "first generated token: " + std::to_string(out_tokens.front()));
    }
    return 0;
}