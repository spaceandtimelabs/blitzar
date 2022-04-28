#pragma once

#include <cstring>
#include <cstdint>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// load64_le
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline uint64_t load64_le(const uint8_t src[8]) noexcept {
  // note: assume the architecture is little endian
  uint64_t res;
  std::memcpy(static_cast<void*>(&res), static_cast<const void*>(src),
              sizeof(res));
  return res;
}
} // namespace sxt::basbt
