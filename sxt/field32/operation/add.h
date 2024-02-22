/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2023
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/field32/type/element.h"

namespace sxt::f32o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/*
 h = f + g
 Can overlap h with f or g.
 */
template <class T1, class T2, class T3>
CUDA_CALLABLE inline void add(T1& h, const T2& f, const T3& g) noexcept {
  uint32_t h0 = f[0] + g[0];
  uint32_t h1 = f[1] + g[1];
  uint32_t h2 = f[2] + g[2];
  uint32_t h3 = f[3] + g[3];
  uint32_t h4 = f[4] + g[4];
  uint32_t h5 = f[5] + g[5];
  uint32_t h6 = f[6] + g[6];
  uint32_t h7 = f[7] + g[7];
  uint32_t h8 = f[8] + g[8];
  uint32_t h9 = f[9] + g[9];

  h[0] = h0;
  h[1] = h1;
  h[2] = h2;
  h[3] = h3;
  h[4] = h4;
  h[5] = h5;
  h[6] = h6;
  h[7] = h7;
  h[8] = h8;
  h[9] = h9;
}
} // namespace sxt::f32o
