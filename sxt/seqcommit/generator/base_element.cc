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
#include "sxt/seqcommit/generator/base_element.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/random/element.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// compute_base_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void compute_base_element(c21t::element_p3& g, uint64_t index) noexcept {
  // Note: we'll probably substitute a different generator in the future, but
  // this works as a placeholder for now
  basn::fast_random_number_generator rng{index + 1, index + 2};
  rstrn::generate_random_element(g, rng);
}

//--------------------------------------------------------------------------------------------------
// compute_compressed_base_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void compute_compressed_base_element(rstt::compressed_element& g_rt, uint64_t index) noexcept {
  c21t::element_p3 g;
  compute_base_element(g, index);
  rstb::to_bytes(g_rt.data(), g);
}
} // namespace sxt::sqcgn
