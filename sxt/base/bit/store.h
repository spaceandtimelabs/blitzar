#pragma once

#include <cstring>
#include <cstdint>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// store64_le
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void store64_le(uint8_t dst[8], uint64_t w) noexcept {
  // note: assume the architecture is little endian
  std::memcpy(static_cast<void*>(dst), static_cast<const void*>(&w), sizeof(w));
}
} // namespace sxt::basbt
