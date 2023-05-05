#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// neg
//--------------------------------------------------------------------------------------------------
/* r = -p */
CUDA_CALLABLE
void neg(c21t::element_p3& r, const c21t::element_p3& p) noexcept;

//--------------------------------------------------------------------------------------------------
// cneg
//--------------------------------------------------------------------------------------------------
/* r = -r if b = 1 else r */
CUDA_CALLABLE
void cneg(c21t::element_p3& r, unsigned int b) noexcept;
} // namespace sxt::c21o
