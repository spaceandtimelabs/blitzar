#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// sqmul
//--------------------------------------------------------------------------------------------------
// Input:
//   a[0]+256*a[1]+...+256^31*a[31] = a
//   n
//
// Output:
//   s[0]+256*s[1]+...+256^31*s[31] = a * s^(2^n) mod l
//
// where l = 2^252 + 27742317777372353535851937790883648493.
//
// Overwrites s in place.
CUDA_CALLABLE
void sqmul(s25t::element& s, const uint32_t n, const s25t::element& a) noexcept;
} // namespace sxt::s25o
