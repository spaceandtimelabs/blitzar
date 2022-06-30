#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxb { struct exponent_sequence; }

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// compute_uint64_muladd
//--------------------------------------------------------------------------------------------------
void compute_uint64_muladd(
    basct::span<uint64_t> result,
    basct::span<uint64_t> generators,
    basct::span<mtxb::exponent_sequence> sequences
) noexcept;
}  // namespace sxt::mtxtst
