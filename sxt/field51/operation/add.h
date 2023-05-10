/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
template <class T1, class T2, class T3>
CUDA_CALLABLE inline void add(T1& h, const T2& f, const T3& g) noexcept {
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
} // namespace sxt::f51o
