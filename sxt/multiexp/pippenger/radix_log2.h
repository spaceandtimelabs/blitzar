#pragma once

#include <cstddef>

namespace sxt::mtxb { class exponent; }

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// compute_radix_log2
//--------------------------------------------------------------------------------------------------
size_t compute_radix_log2(const sxt::mtxb::exponent& max_exponent, size_t num_inputs, size_t num_outputs) noexcept;

}  // namespace sxt::mtxpi
