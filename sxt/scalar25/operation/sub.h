#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// sub
//--------------------------------------------------------------------------------------------------
//
// Input:
//   a[0]+256*a[1]+...+256^31*a[31] = a
//   b[0]+256*b[1]+...+256^31*b[31] = b
//
// Output:
//   s[0]+256*s[1]+...+256^31*s[31] = (a - b) mod l
//
// where l = 2^252 + 27742317777372353535851937790883648493
CUDA_CALLABLE
void sub(s25t::element& s, const s25t::element& a, const s25t::element& b) noexcept;
} // namespace sxt::s25o
