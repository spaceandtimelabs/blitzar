#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// neg
//--------------------------------------------------------------------------------------------------
//
// Input:
//   a[0]+256*a[1]+...+256^31*a[31] = s
//
// Output:
//   n where (s + n) % l = 0
//
// s = s[0]+256*s[1]+...+256^31*s[31]
//
// where l = 2^252 + 27742317777372353535851937790883648493
CUDA_CALLABLE
void neg(s25t::element& n, const s25t::element& s) noexcept;
} // namespace sxt::s25o
