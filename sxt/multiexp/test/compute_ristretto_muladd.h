#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::rstt {
struct compressed_element;
}

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// compute_ristretto_muladd
//--------------------------------------------------------------------------------------------------
void compute_ristretto_muladd(basct::span<rstt::compressed_element> result,
                              basct::span<rstt::compressed_element> generators,
                              basct::span<mtxb::exponent_sequence> sequences) noexcept;
} // namespace sxt::mtxtst
