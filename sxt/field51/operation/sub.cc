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
#include "sxt/field51/operation/sub.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// sub
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void sub(f51t::element& h, const f51t::element& f, const f51t::element& g) noexcept {
  const uint64_t mask = 0x7ffffffffffffULL;
  uint64_t h0, h1, h2, h3, h4;

  h0 = g[0];
  h1 = g[1];
  h2 = g[2];
  h3 = g[3];
  h4 = g[4];

  h1 += h0 >> 51;
  h0 &= mask;
  h2 += h1 >> 51;
  h1 &= mask;
  h3 += h2 >> 51;
  h2 &= mask;
  h4 += h3 >> 51;
  h3 &= mask;
  h0 += 19ULL * (h4 >> 51);
  h4 &= mask;

  h0 = (f[0] + 0xfffffffffffdaULL) - h0;
  h1 = (f[1] + 0xffffffffffffeULL) - h1;
  h2 = (f[2] + 0xffffffffffffeULL) - h2;
  h3 = (f[3] + 0xffffffffffffeULL) - h3;
  h4 = (f[4] + 0xffffffffffffeULL) - h4;

  h[0] = h0;
  h[1] = h1;
  h[2] = h2;
  h[3] = h3;
  h[4] = h4;
}
} // namespace sxt::f51o
