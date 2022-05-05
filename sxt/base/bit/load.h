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

CUDA_CALLABLE
inline uint64_t load_3(const unsigned char *in) {
    uint64_t result;

    result = (uint64_t) in[0];
    result |= ((uint64_t) in[1]) << 8;
    result |= ((uint64_t) in[2]) << 16;

    return result;
}

CUDA_CALLABLE
inline uint64_t load_4(const unsigned char *in) {
    uint64_t result;

    result = (uint64_t) in[0];
    result |= ((uint64_t) in[1]) << 8;
    result |= ((uint64_t) in[2]) << 16;
    result |= ((uint64_t) in[3]) << 24;

    return result;
}

} // namespace sxt::basbt
