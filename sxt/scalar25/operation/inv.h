#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// inv
//--------------------------------------------------------------------------------------------------
//
// Input:
//   a[0]+256*a[1]+...+256^31*a[31] = s
//
// Output:
//   s_inv = s_inv[0]+256*s_inv[1]+...+256^31*s_inv[31] where (s * s_inv) = 1 % l
//
// where l = 2^252 + 27742317777372353535851937790883648493
CUDA_CALLABLE
void inv(s25t::element& s_inv, const s25t::element& s) noexcept;
} // namespace sxt::s25o
