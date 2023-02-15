#pragma once

#include "sxt/base/container/span.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
memmg::managed_array<c21t::element_p3>
compute_multiexponentiation(basct::cspan<c21t::element_p3> generators,
                            basct::cspan<mtxb::exponent_sequence> exponents) noexcept;
} // namespace sxt::mtxc21
