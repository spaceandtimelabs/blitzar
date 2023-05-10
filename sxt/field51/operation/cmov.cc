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
#include "sxt/field51/operation/cmov.h"

namespace sxt::f51o {
/*
 Replace (f,g) with (g,g) if b == 1;
 replace (f,g) with (f,g) if b == 0.
 *
 Preconditions: b in {0,1}.
 */
CUDA_CALLABLE void cmov(f51t::element& f, const f51t::element& g, unsigned int b) noexcept {
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
} // namespace sxt::f51o
