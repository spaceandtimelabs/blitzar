#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t { struct element_p3; }
namespace sxt::f51t { class element; }

namespace sxt::c21rs {
//--------------------------------------------------------------------------------------------------
// apply_elligator
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void apply_elligator(c21t::element_p3& p, const f51t::element& t) noexcept;
}  // namespace sxt::c21rs
