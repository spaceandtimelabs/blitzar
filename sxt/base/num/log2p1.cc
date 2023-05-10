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
#include "sxt/base/num/log2p1.h"

#include <cmath>

#include "sxt/base/error/assert.h"

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// log2p1
//--------------------------------------------------------------------------------------------------
double log2p1(basct::cspan<uint8_t> x) noexcept {
  double res = 1;
  double power256 = 1.;

  // only numbers smaller than 127 bytes (1016 bits) are allowed
  SXT_DEBUG_ASSERT(x.size() <= 127);

  for (size_t i = 0; i < x.size(); ++i) {
    res += static_cast<double>(x[i]) * power256;
    power256 *= 256.;
  }

  return std::log2(res);
}
} // namespace sxt::basn
