#pragma once

#include "sxt/base/container/span.h"

#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxpi {
class driver;

//--------------------------------------------------------------------------------------------------
// compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
void compute_multiexponentiation(memmg::managed_array<void>& inout, const driver& drv,
                                 basct::cspan<mtxb::exponent_sequence> exponents) noexcept;
} // namespace sxt::mtxpi
