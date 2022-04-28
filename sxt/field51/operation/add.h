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
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/*
 h = f + g
 Can overlap h with f or g.
 */
CUDA_CALLABLE
inline void add(f51t::element& h, const f51t::element& f, const f51t::element& g) noexcept {
  uint64_t h0 = f[0] + g[0];
  uint64_t h1 = f[1] + g[1];
  uint64_t h2 = f[2] + g[2];
  uint64_t h3 = f[3] + g[3];
  uint64_t h4 = f[4] + g[4];

  h[0] = h0;
  h[1] = h1;
  h[2] = h2;
  h[3] = h3;
  h[4] = h4;
}
}  // namespace sxt::f51o
