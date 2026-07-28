// Quantized weight format registry helpers.
#include "viper/quant.h"

namespace viper::quant {

std::string_view qtype_name(QType q) noexcept {
    switch (q) {
        case QType::BF16:   return "BF16";
        case QType::FP32:   return "FP32";
        case QType::Q4_G64: return "Q4_G64";
        case QType::Q5_G64: return "Q5_G64";
        case QType::Q6_G64: return "Q6_G64";
        case QType::W8_G32: return "W8_G32";
    }
    return "UNKNOWN";
}

bool is_lossless(QType q) noexcept {
    return q == QType::BF16 || q == QType::FP32;
}

}  // namespace viper::quant