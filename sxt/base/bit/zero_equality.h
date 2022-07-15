/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#pragma once

#include <cstddef>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// is_zero
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline int is_zero(const unsigned char* n, const size_t nlen) noexcept {
  size_t i;
  volatile unsigned char d = 0U;

  for (i = 0U; i < nlen; i++) {
    d |= n[i];
  }
  return 1 & ((d - 1) >> 8);
}
} // namespace sxt::basbt
