#pragma once

#include "sxt/base/macro/cuda_callable.h"

#include "sxt/field51/base/byte_conversion.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51p {
//--------------------------------------------------------------------------------------------------
// is_negative
//--------------------------------------------------------------------------------------------------
/*
 return 1 if f is in {1,3,5,...,q-2}
 return 0 if f is in {0,2,4,...,q-1}
 */
CUDA_CALLABLE
inline int is_negative(const f51t::element& f) noexcept {
  unsigned char s[32];

  f51b::to_bytes(s, f.data());

  return s[0] & 1;
}
} // namespace sxt::f51p
