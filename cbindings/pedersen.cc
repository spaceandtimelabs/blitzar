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
#include "cbindings/pedersen.h"

#include <iostream>

#include "cbindings/backend.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve_g1/type/compressed_element.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/ristretto/type/compressed_element.h"

using namespace sxt;

namespace sxt::cbn {
//--------------------------------------------------------------------------------------------------
// populate_exponent_sequence
//--------------------------------------------------------------------------------------------------
static uint64_t populate_exponent_sequence(basct::span<mtxb::exponent_sequence> sequences,
                                           basct::cspan<sxt_sequence_descriptor> descriptors) {
  SXT_RELEASE_ASSERT(descriptors.data() != nullptr);

  uint64_t longest_sequence = 0;

  for (uint32_t i = 0; i < sequences.size(); ++i) {
    auto& curr_descriptor = descriptors[i];

    SXT_RELEASE_ASSERT(curr_descriptor.n == 0 || curr_descriptor.data != nullptr);

    SXT_RELEASE_ASSERT(curr_descriptor.element_nbytes != 0 && curr_descriptor.element_nbytes <= 32);

    longest_sequence = std::max(longest_sequence, curr_descriptor.n);

    sequences[i] = {
        .element_nbytes = curr_descriptor.element_nbytes,
        .n = curr_descriptor.n,
        .data = curr_descriptor.data,
        .is_signed = curr_descriptor.is_signed,
    };
  }

  return longest_sequence;
}

//--------------------------------------------------------------------------------------------------
// process_compute_pedersen_commitments
//--------------------------------------------------------------------------------------------------
static void process_compute_pedersen_commitments(struct sxt_ristretto255_compressed* commitments,
                                                 basct::cspan<sxt_sequence_descriptor> descriptors,
                                                 const c21t::element_p3* generators,
                                                 uint64_t offset_generators) {
  if (descriptors.size() == 0)
    return;

  SXT_RELEASE_ASSERT(commitments != nullptr);
  SXT_RELEASE_ASSERT(sxt::cbn::is_backend_initialized());
  static_assert(sizeof(rstt::compressed_element) == sizeof(sxt_ristretto255_compressed),
                "types must be ABI compatible");

  memmg::managed_array<mtxb::exponent_sequence> sequences(descriptors.size());
  auto num_generators = populate_exponent_sequence(sequences, descriptors);

  auto backend = cbn::get_backend();
  std::vector<c21t::element_p3> temp_generators;
  basct::cspan<c21t::element_p3> generators_span;

  if (generators == nullptr) {
    generators_span =
        backend->get_precomputed_generators(temp_generators, num_generators, offset_generators);
  } else {
    generators_span = basct::cspan<c21t::element_p3>(generators, num_generators);
  }

  backend->compute_commitments(
      {reinterpret_cast<rstt::compressed_element*>(commitments), descriptors.size()}, sequences,
      generators_span);
}

//--------------------------------------------------------------------------------------------------
// process_compute_pedersen_commitments_bls12_381_g1
//--------------------------------------------------------------------------------------------------
static void process_compute_pedersen_commitments_bls12_381_g1(
    struct sxt_compressed_g1* commitments, basct::cspan<sxt_sequence_descriptor> descriptors,
    const cg1t::element_p2* generators, uint64_t offset_generators) {
  if (descriptors.size() == 0)
    return;

  SXT_RELEASE_ASSERT(commitments != nullptr);
  SXT_RELEASE_ASSERT(generators == nullptr);
  SXT_RELEASE_ASSERT(sxt::cbn::is_backend_initialized());
  static_assert(sizeof(cg1t::compressed_element) == sizeof(sxt_compressed_g1),
                "types must be ABI compatible");

  memmg::managed_array<mtxb::exponent_sequence> sequences(descriptors.size());
  auto num_generators = populate_exponent_sequence(sequences, descriptors);

  auto backend = cbn::get_backend();
  std::vector<cg1t::element_p2> temp_generators;
  basct::cspan<cg1t::element_p2> generators_span{generators, num_generators};

  backend->compute_commitments(
      {reinterpret_cast<cg1t::compressed_element*>(commitments), descriptors.size()}, sequences,
      generators_span);
}

//--------------------------------------------------------------------------------------------------
// process_compute_pedersen_commitments_bls12_381_g1
//--------------------------------------------------------------------------------------------------
static void process_compute_pedersen_commitments_bls12_381_g1(
    struct sxt_compressed_g1* commitments, basct::cspan<sxt_sequence_descriptor> descriptors,
    const cg1t::element_p2* generators, uint64_t offset_generators) {
  if (descriptors.size() == 0)
    return;

  SXT_RELEASE_ASSERT(commitments != nullptr);
  SXT_RELEASE_ASSERT(generators == nullptr);
  SXT_RELEASE_ASSERT(sxt::cbn::is_backend_initialized());
  static_assert(sizeof(cg1t::compressed_element) == sizeof(sxt_compressed_g1),
                "types must be ABI compatible");

  memmg::managed_array<mtxb::exponent_sequence> sequences(descriptors.size());
  auto num_generators = populate_exponent_sequence(sequences, descriptors);

  auto backend = cbn::get_backend();
  std::vector<cg1t::element_p2> temp_generators;
  basct::cspan<cg1t::element_p2> generators_span{generators, num_generators};

  backend->compute_commitments(
      {reinterpret_cast<cg1t::compressed_element*>(commitments), descriptors.size()}, sequences,
      generators_span);
}
} // namespace sxt::cbn

//--------------------------------------------------------------------------------------------------
// sxt_curve25519_compute_pedersen_commitments_with_generators
//--------------------------------------------------------------------------------------------------
void sxt_curve25519_compute_pedersen_commitments_with_generators(
    struct sxt_ristretto255_compressed* commitments, uint32_t num_sequences,
    const struct sxt_sequence_descriptor* descriptors, const struct sxt_ristretto255* generators) {
  cbn::process_compute_pedersen_commitments(commitments, {descriptors, num_sequences},
                                            reinterpret_cast<const c21t::element_p3*>(generators),
                                            0);
}

//--------------------------------------------------------------------------------------------------
// sxt_compute_pedersen_commitments_with_bls12_381_g1_generators
//--------------------------------------------------------------------------------------------------
void sxt_compute_pedersen_commitments_with_bls12_381_g1_generators(
    struct sxt_compressed_g1* commitments, uint32_t num_sequences,
    const struct sxt_sequence_descriptor* descriptors, const struct sxt_g1* generators) {
  cbn::process_compute_pedersen_commitments_bls12_381_g1(
      commitments, {descriptors, num_sequences},
      reinterpret_cast<const cg1t::element_p2*>(generators), 0);
}

//--------------------------------------------------------------------------------------------------
// sxt_compute_pedersen_commitments
//--------------------------------------------------------------------------------------------------
void sxt_curve25519_compute_pedersen_commitments(sxt_ristretto255_compressed* commitments,
                                                 uint32_t num_sequences,
                                                 const sxt_sequence_descriptor* descriptors,
                                                 uint64_t offset_generators) {
  cbn::process_compute_pedersen_commitments(commitments, {descriptors, num_sequences}, nullptr,
                                            offset_generators);
}
