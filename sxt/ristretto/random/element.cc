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
#include "sxt/ristretto/random/element.h"

#include "sxt/base/macro/cuda_warning.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve32/type/element_p3.h"
#include "sxt/field32/random/element.h"
#include "sxt/field32/type/element.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/base/point_formation.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::rstrn {
//--------------------------------------------------------------------------------------------------
// generate_random_element_impl
//--------------------------------------------------------------------------------------------------
CUDA_DISABLE_HOSTDEV_WARNING
template <class Rng>
static CUDA_CALLABLE void generate_random_element_impl(c32t::element_p3& p, Rng& rng) noexcept {
  f32t::element r0, r1;
  f32rn::generate_random_element(r0, rng);
  f32rn::generate_random_element(r1, rng);
  rstb::form_ristretto_point(p, r0, r1);
}

//--------------------------------------------------------------------------------------------------
// generate_random_elements_impl
//--------------------------------------------------------------------------------------------------
template <class Rng>
static void generate_random_elements_impl(basct::span<c32t::element_p3> px, Rng& rng) noexcept {
  for (auto& pi : px) {
    generate_random_element(pi, rng);
  }
}

//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void generate_random_element(c32t::element_p3& p,
                             basn::fast_random_number_generator& rng) noexcept {
  generate_random_element_impl(p, rng);
}

void generate_random_element(c32t::element_p3& p, std::mt19937& rng) noexcept {
  generate_random_element_impl(p, rng);
}

void generate_random_element(rstt::compressed_element& p,
                             basn::fast_random_number_generator& rng) noexcept {
  c32t::element_p3 pp;
  generate_random_element(pp, rng);
  rstb::to_bytes(p.data(), pp);
}

//--------------------------------------------------------------------------------------------------
// generate_random_elements
//--------------------------------------------------------------------------------------------------
void generate_random_elements(basct::span<c32t::element_p3> px,
                              basn::fast_random_number_generator& rng) noexcept {
  generate_random_elements_impl(px, rng);
}

void generate_random_elements(basct::span<c32t::element_p3> px, std::mt19937& rng) noexcept {
  generate_random_elements_impl(px, rng);
}

void generate_random_elements(basct::span<rstt::compressed_element> px,
                              basn::fast_random_number_generator& rng) noexcept {
  for (auto& pi : px) {
    generate_random_element(pi, rng);
  }
}
} // namespace sxt::rstrn
