#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::f51t { class element; }

namespace sxt::c21rs {
//--------------------------------------------------------------------------------------------------
// compute_sqrt_ratio_m1
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
int compute_sqrt_ratio_m1(f51t::element& x, const f51t::element& u,
                          const f51t::element& v) noexcept;
}  // namespace sxt::c21rs
