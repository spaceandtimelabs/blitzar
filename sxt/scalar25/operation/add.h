#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
//
// Input:
//   x[0]+256*x[1]+...+256^31*x[31] = x
//   y[0]+256*y[1]+...+256^31*y[31] = y
//
// Output:
//   z[0]+256*z[1]+...+256^31*z[31] = (x + y) mod l
//
// where l = 2^252 + 27742317777372353535851937790883648493
CUDA_CALLABLE
void add(s25t::element& z, const s25t::element& x, const s25t::element& y) noexcept;

CUDA_CALLABLE
void add(s25t::element& z, const volatile s25t::element& x,
         const volatile s25t::element& y) noexcept;
} // namespace sxt::s25o
