/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51o {
/*
 Replace (f,g) with (g,g) if b == 1;
 replace (f,g) with (f,g) if b == 0.
 *
 Preconditions: b in {0,1}.
 */
CUDA_CALLABLE inline void cmov(f51t::element& f, const f51t::element& g,
                               unsigned int b) noexcept {
  uint64_t mask = (uint64_t)(-(int64_t)b);
  uint64_t f0, f1, f2, f3, f4;
  uint64_t x0, x1, x2, x3, x4;

  f0 = f[0];
  f1 = f[1];
  f2 = f[2];
  f3 = f[3];
  f4 = f[4];

  x0 = f0 ^ g[0];
  x1 = f1 ^ g[1];
  x2 = f2 ^ g[2];
  x3 = f3 ^ g[3];
  x4 = f4 ^ g[4];

  x0 &= mask;
  x1 &= mask;
  x2 &= mask;
  x3 &= mask;
  x4 &= mask;

  f[0] = f0 ^ x0;
  f[1] = f1 ^ x1;
  f[2] = f2 ^ x2;
  f[3] = f3 ^ x3;
  f[4] = f4 ^ x4;
}
}  // namespace sxt::f51o
