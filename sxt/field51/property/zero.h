#pragma once

#include "sxt/base/bit/zero_equality.h"
#include "sxt/base/macro/cuda_callable.h"

#include "sxt/field51/base/byte_conversion.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51p {
//--------------------------------------------------------------------------------------------------
// is_zero
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline int is_zero(const f51t::element& e) noexcept {
  unsigned char bytes[32];
  f51b::to_bytes(bytes, e.data());
  return basbt::is_zero(bytes, sizeof(bytes));
}
} // namespace sxt::f51p
