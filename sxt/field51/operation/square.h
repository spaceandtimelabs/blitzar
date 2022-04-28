#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::f51t { class element; }

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// square
//--------------------------------------------------------------------------------------------------
/*
 h = f * f
 Can overlap h with f.
 */

CUDA_CALLABLE
void square(f51t::element& h, const f51t::element& f) noexcept;

//--------------------------------------------------------------------------------------------------
// square2
//--------------------------------------------------------------------------------------------------
/*
 h = 2 * f * f
 Can overlap h with f.
*/
CUDA_CALLABLE
void square2(f51t::element& h, const f51t::element& f) noexcept;
}  // namespace sxt::f51o
