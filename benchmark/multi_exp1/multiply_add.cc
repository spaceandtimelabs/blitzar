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
#include "benchmark/multi_exp1/multiply_add.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/random/exponent.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/random/element.h"

namespace sxt {
//--------------------------------------------------------------------------------------------------
// multiply_add
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void multiply_add(c21t::element_p3& res, int mi, int i) noexcept {
  basn::fast_random_number_generator rng{static_cast<uint64_t>(i + 1),
                                         static_cast<uint64_t>(mi + 1)};

  // pretend like g is a random element rather than fixed
  c21t::element_p3 g;
  rstrn::generate_random_element(g, rng);

  unsigned char a[32];
  c21rn::generate_random_exponent(a, rng);
  c21t::element_p3 e;
  c21o::scalar_multiply255(e, a, g);
  c21o::add(res, res, e);
}
} // namespace sxt
