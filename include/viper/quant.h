// Closed weight format registry. linear() is a switch on w.qtype — never a
// generic backend. Adding a new format means editing this header and the
// switch in ops.cpp.
//
// Lossless (default, bit-exact to BF16 reference):
//   BF16, FP32
// Lossy (off by default, must be validated against FP64 Python oracle before
// use; CLI prints a quality-delta warning):
//   Q4_G64, Q5_G64, Q6_G64, W8_G32
#pragma once

#include <string_view>

#include "viper/common.h"
#include "viper/tensor.h"

namespace viper::quant {

enum class QType : int {
    BF16 = 0,    // lossless
    FP32 = 1,    // lossless
    Q4_G64 = 4,  // lossy
    Q5_G64 = 5,  // lossy
    Q6_G64 = 6,  // lossy
    W8_G32 = 8,  // lossy
};

std::string_view qtype_name(QType q) noexcept;
bool is_lossless(QType q) noexcept;

// QuantizedWeight — opaque view over a packed buffer + per-block scales.
// layout details (block size, scale dtype, etc.) live in the per-format
// decoder; consumers go through ops::linear_forward and never touch layout.
struct QuantizedWeight {
    QType qtype = QType::BF16;
    void* data = nullptr;       // device pointer to packed weights
    void* scales = nullptr;     // device pointer to scales (or nullptr for fp)
    i32 rows = 0;               // output channels
    i32 cols = 0;               // input channels
    i32 group_size = 64;        // quant group size (64 for Q*_G64, 32 for W8_G32)
};

// Lossy-vs-lossless delta (printed at startup when lossy opts are enabled).
struct QualityDelta {
    f64 max_abs_err = 0.0;
    f64 mean_abs_err = 0.0;
    f64 kl_divergence = 0.0;  // vs FP64 oracle logits
    i32 oracle_vocab = 166144;
    i32 oracle_tokens = 0;
};

}  // namespace viper::quant