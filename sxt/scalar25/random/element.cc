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
#include "sxt/scalar25/random/element.h"

#include <cstring>

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/scalar25/base/reduce.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25rn {
//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
void generate_random_element(s25t::element& e, basn::fast_random_number_generator& rng) noexcept {
  auto data = e.data();
  for (int i = 0; i < 32; i += 8) {
    auto x = rng();
    std::memcpy(static_cast<void*>(data + i), static_cast<void*>(&x), sizeof(x));
  }
  s25b::reduce32(data);
}

//--------------------------------------------------------------------------------------------------
// generate_random_elements
//--------------------------------------------------------------------------------------------------
void generate_random_elements(basct::span<s25t::element> ex,
                              basn::fast_random_number_generator& rng) noexcept {
  for (auto& ei : ex) {
    generate_random_element(ei, rng);
  }
}
} // namespace sxt::s25rn
