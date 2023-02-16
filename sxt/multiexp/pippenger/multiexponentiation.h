#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_void.h"
#include "sxt/execution/async/future_fwd.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxpi {
class driver;

//--------------------------------------------------------------------------------------------------
// compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<void>>
compute_multiexponentiation(const driver& drv, basct::span_cvoid generators,
                            basct::cspan<mtxb::exponent_sequence> exponents) noexcept;
} // namespace sxt::mtxpi
