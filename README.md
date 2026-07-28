# Viper Inference Engine

C++/CUDA inference engine for **Nanbeige4.2-3B** targeting **RTX 3070 Ti Laptop** (sm_86, 8 GB, 448 GB/s).
Target: 500-600+ tok/s decode, 262 144 max context window.

## Design contract

- **Megakernel-first.** A single persistent kernel runs the full 44-step forward pass and grid-syncs between layers via cooperative groups. CUDA Graphs are **forbidden** anywhere in the engine — they destabilize the GPU.
- **No CUDA Graphs.** No `cudaGraph*`, `cudaStreamBeginCapture`, or `cudaGraphLaunch` calls. The persistent megakernel saves the launch overhead CUDA Graphs would have captured.
- **Closed weight format registry.** `BF16 | FP32 | Q4_G64 | Q5_G64 | Q6_G64 | W8_G32`. `linear()` is a switch on `w.qtype` — never a generic backend.
- **44-slot KV cache.** `cache_layer_idx = loop_idx * 22 + layer_idx`. position_ids shared across both passes; inter-pass RMSNorm between loops (`skip_loop_final_norm=False`).
- **Lossless default.** Default config is bit-exact to BF16 reference. Lossy opts (Q4 weights, INT8 KV, W8_G32) are **off** by default and gated behind `--quality-lossy` with a printed quality-delta warning.
- **Safety guards everywhere.** `cudaMemGetInfo` before every major alloc (refuse < 1 GiB free), host RAM guard (refuse < 4 GiB free), per-op timeout (60 s default), NVML temp check (warn 80 °C, abort 87 °C), all public APIs return `Status`.

## Layout

```
D:\dev\viper\
├── CMakeLists.txt              # root CMake, sm_86 only
├── include\viper\              # public headers
│   ├── common.h                # scalar types, logging, VIPER_CHECK
│   ├── status.h                # Status / StatusCode
│   ├── cuda_check.h            # VIPER_CHECK_CUDA + classify_cuda
│   ├── tensor.h                # Tensor / Shape / DeviceBuffer
│   ├── ops.h                   # op signatures (rmsnorm, rope, …)
│   ├── kv_cache.h              # 44-slot cache
│   ├── model.h                 # NanbeigeModel
│   ├── quant.h                 # closed format registry
│   ├── megakernel.h            # persistent kernel launcher (NO CUDA Graphs)
│   └── safety.h                # resource / temp / cuda error guards
├── src\viper\                  # host implementations
│   ├── common.cpp              # logging
│   ├── tensor.cpp              # DeviceBuffer + dtype
│   ├── safety.cpp              # cudaMemGetInfo + NVML + host RAM
│   ├── kv_cache.cpp            # 44-slot cache manager
│   ├── model.cpp               # NanbeigeModel impl
│   ├── quant.cpp               # format registry helpers
│   └── <op>.cpp                # one per op — stub returning UNIMPLEMENTED
│       (rmsnorm, rope, embedding, linear,
│        gqa_repeat, sdpa, swiglu, residual, sampling)
├── kernels\ops\                # main thread owns (REAL CUDA kernels)
│   ├── rmsnorm_kernel.h/.cu    # main
│   ├── rope_kernel.h/.cu       # main
│   ├── embedding_kernel.h/.cu  # main
│   ├── swiglu_kernel.h/.cu     # main
│   ├── residual_kernel.h/.cu   # main
│   └── …                       # more landing soon
├── tools\
│   ├── cli\main.cpp            # viper_cli
│   ├── serve\main.cpp          # viper_serve (HTTP placeholder, :8080)
│   └── convert\main.cpp        # viper_convert (M1 placeholder)
├── tests\                      # main thread owns (tests/*.cu)
└── scripts\
    ├── build.bat               # MSVC + CUDA + CMake wrapper
    └── build.ps1               # PowerShell variant
```

## Build

Toolchain: CUDA 12.8, MSVC 2019 14.29, CMake 4.1, Ninja or NMake.

```bat
scripts\build.bat
```

Configurable:

```bat
scripts\build.bat RelWithDebInfo Ninja
```

Outputs land in `build\bin\`:

- `viper_cli.exe`     — loads a `.viper` artifact, prints one token
- `viper_serve.exe`   — placeholder HTTP server, binds 127.0.0.1:8080 (returns 501)
- `viper_convert.exe` — placeholder format converter (M1)

## Status matrix (skeleton phase)

| Component | Status |
|-----------|--------|
| `common.h`, `tensor.h`, `safety.h`, `status.h`, `cuda_check.h` | real |
| `megakernel::launch_persistent_forward` | stub (UNIMPLEMENTED — main thread wires M3) |
| `kv_cache`, `model`, `quant` | real interface + placeholder parser |
| All ops (`rmsnorm`, `rope`, `embedding`, `linear`, `sdpa`, `swiglu`, `residual`, `sampling`, `gqa_repeat`) | stub `Status::UNIMPLEMENTED`; real kernels land in `kernels/ops/*.cu` |
| `viper_serve`           | placeholder; binds 8080; returns 501 |
| `viper_convert`         | placeholder |
| `NanbeigeModel::forward`| stub |

## CLI flags

```
viper_cli --load <path> [--prompt <text>]
           [--max-iters N] [--max-tokens N] [--max-memory MB]
           [--quality-lossy]
           [--help]
```

`--quality-lossy` opts into Q4_G64 / Q5_G64 / Q6_G64 / W8_G32 and prints a quality-delta warning vs the BF16 oracle.