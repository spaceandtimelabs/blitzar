#include "sxt/multiexp/pippenger/radix_log2.h"

#include <cmath>

#include "sxt/base/num/log2p1.h"
#include "sxt/base/container/span.h"
#include "sxt/multiexp/base/exponent.h"

#include <stdio.h>

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// compute_radix_log2
//--------------------------------------------------------------------------------------------------
size_t compute_radix_log2(
    const sxt::mtxb::exponent& max_exponent,
    size_t num_inputs, size_t num_outputs) noexcept {

    // check if any of the input values are zero (including the max_exponent)
    if (num_outputs == 0 || num_inputs == 0 ||
            max_exponent == sxt::mtxb::exponent()) return 1;

    double multiplier_factor;

    if (num_inputs >= num_outputs) {
        multiplier_factor = static_cast<double>(num_outputs) / num_inputs;
    } else {
        multiplier_factor = static_cast<double>(num_inputs) / num_outputs;
    }

    double log_val = sxt::basn::log2p1(sxt::basct::cspan<uint8_t>(
        reinterpret_cast<const uint8_t *>(max_exponent.data()), sizeof(max_exponent)
    ));

    return std::max(1lu, static_cast<size_t>(std::ceil(std::sqrt(multiplier_factor * log_val))));
}
}
