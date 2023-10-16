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

#include <type_traits>
#include <vector>

#include "cbindings/backend.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/operation/add.h"
#include "sxt/ristretto/operation/overload.h"
#include "sxt/ristretto/operation/scalar_multiply.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/generator/base_element.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// initialize_backend
//--------------------------------------------------------------------------------------------------
static void initialize_backend(int backend, uint64_t precomputed_elements) {
  const sxt_config config = {backend, precomputed_elements};
  REQUIRE(sxt_init(&config) == 0);
}

//--------------------------------------------------------------------------------------------------
// compute_random_curve25519_generators
//--------------------------------------------------------------------------------------------------
static std::vector<c21t::element_p3> compute_random_curve25519_generators(uint64_t seq_length,
                                                                          uint64_t offset) {
  std::vector<c21t::element_p3> generators(seq_length);

  for (uint64_t i = 0; i < seq_length; ++i) {
    sqcgn::compute_base_element(generators[i], offset + i);
  }

  return generators;
}

//--------------------------------------------------------------------------------------------------
// make_sequence_descriptor
//--------------------------------------------------------------------------------------------------
template <class T>
static sxt_sequence_descriptor make_sequence_descriptor(const std::vector<T>& data) {
  return {
      .element_nbytes = sizeof(T),
      .n = data.size(),
      .data = reinterpret_cast<const uint8_t*>(data.data()),
      .is_signed = std::is_signed_v<T>,
  };
}

//--------------------------------------------------------------------------------------------------
// compute_expected_ristretto255_commitment
//--------------------------------------------------------------------------------------------------
template <class T>
static rstt::compressed_element
compute_expected_ristretto255_commitment(const std::vector<T>& data,
                                         const std::vector<c21t::element_p3>& generators) {
  SXT_DEBUG_ASSERT(data.size() == generators.size());

  rstt::compressed_element expected_commitment;
  rstb::to_bytes(expected_commitment.data(), c21t::element_p3::identity());

  for (uint64_t i = 0; i < data.size(); ++i) {
    rstt::compressed_element aux_h;
    rstb::to_bytes(aux_h.data(), generators[i]);
    rsto::scalar_multiply(aux_h, data[i], aux_h);
    rsto::add(expected_commitment, expected_commitment, aux_h);
  }

  return expected_commitment;
}

//--------------------------------------------------------------------------------------------------
// test_ristretto255_pedersen_commitments_with_given_backend_and_no_generators
//--------------------------------------------------------------------------------------------------
static void test_ristretto255_pedersen_commitments_with_given_backend_and_no_generators(
    int backend, uint64_t num_precomputed_generators) {
  initialize_backend(backend, num_precomputed_generators);

  SECTION("Zero num_sequence will not error out even with a null input sequence pointer") {
    const uint64_t offset_gens = 0;
    const uint32_t num_sequences = 0;
    const sxt_sequence_descriptor* invalid_descriptor = nullptr;
    sxt_ristretto255_compressed commitment{1u};
    sxt_curve25519_compute_pedersen_commitments(&commitment, num_sequences, invalid_descriptor,
                                                offset_gens);
    REQUIRE(rstt::compressed_element{1u} ==
            *reinterpret_cast<rstt::compressed_element*>(&commitment));
  }

  SECTION("Input sequences with zero length will not error out even with a null data pointer") {
    const uint64_t offset_gens = 0;
    const std::vector<uint8_t> data(0);
    const auto seq_descriptor = make_sequence_descriptor(data);
    const uint32_t num_sequences = 1;
    sxt_ristretto255_compressed commitment;
    sxt_curve25519_compute_pedersen_commitments(&commitment, num_sequences, &seq_descriptor,
                                                offset_gens);
    REQUIRE(rstt::compressed_element() ==
            *reinterpret_cast<rstt::compressed_element*>(&commitment));
  }

  SECTION("we can compute signed commitments") {
    const std::vector<int64_t> data1 = {-2};
    const std::vector<int64_t> data2 = {2};
    const std::vector<sxt_sequence_descriptor> valid_descriptors = {
        make_sequence_descriptor(data1),
        make_sequence_descriptor(data2),
    };
    const uint64_t num_sequences = valid_descriptors.size();
    rstt::compressed_element commitments_data[num_sequences];
    sxt_curve25519_compute_pedersen_commitments(
        reinterpret_cast<sxt_ristretto255_compressed*>(commitments_data), num_sequences,
        valid_descriptors.data(), 0);
    REQUIRE(commitments_data[0] == -commitments_data[1]);
  }

  SECTION("We can multiply and add two commitments together, then compare them against the c "
          "binding results") {
    const uint64_t scal = 52;
    const uint64_t offset_gens = 0;
    const std::vector<uint64_t> data_1 = {2000, 7500, 5000, 1500};
    const std::vector<uint64_t> data_2 = {5000, 0, 400000, 10};
    const std::vector<uint64_t> data_3 = {
        scal * data_1[0] + data_2[0], scal * data_1[1] + data_2[1], scal * data_1[2] + data_2[2],
        scal * data_1[3] + data_2[3]};
    const std::vector<sxt_sequence_descriptor> valid_descriptors = {
        make_sequence_descriptor(data_1),
        make_sequence_descriptor(data_2),
        make_sequence_descriptor(data_3),
    };
    const uint64_t num_sequences = valid_descriptors.size();

    // we verify that `c = scal * a + b` implies that `commit_c = scal * commit_a + commit_b`
    rstt::compressed_element commitments_data[num_sequences];
    sxt_curve25519_compute_pedersen_commitments(
        reinterpret_cast<sxt_ristretto255_compressed*>(commitments_data), num_sequences,
        valid_descriptors.data(), offset_gens);

    auto commit_a = commitments_data[0], commit_b = commitments_data[1],
         commit_c = commitments_data[0];
    rsto::scalar_multiply(commit_a, scal, commit_a);
    rsto::add(commit_c, commit_a, commit_b);

    REQUIRE(rstt::compressed_element() != commit_c);
    REQUIRE(commit_c == commitments_data[2]);
  }

  SECTION("We can compute commitments with a non-zero offset generator") {
    const uint64_t offset_gens = 10;
    const std::vector<uint8_t> data = {1, 0, 2, 6, 0, 7};
    const auto descriptor = make_sequence_descriptor(data);

    rstt::compressed_element commitments_data;
    sxt_curve25519_compute_pedersen_commitments(
        reinterpret_cast<sxt_ristretto255_compressed*>(&commitments_data), 1, &descriptor,
        offset_gens);

    const auto generators = compute_random_curve25519_generators(data.size(), offset_gens);
    const auto expected_commitment = compute_expected_ristretto255_commitment(data, generators);

    REQUIRE(commitments_data == expected_commitment);
  }

  cbn::reset_backend_for_testing();
}

//--------------------------------------------------------------------------------------------------
// test_ristretto255_pedersen_commitments_with_given_backend_and_generators
//--------------------------------------------------------------------------------------------------
static void test_ristretto255_pedersen_commitments_with_given_backend_and_generators(
    int backend, uint64_t num_precomputed_generators) {
  initialize_backend(backend, num_precomputed_generators);

  SECTION("We verify that using the correct generators will produce correct results") {
    const std::vector<uint32_t> data = {2000, 7500};
    const auto seq_descriptor = make_sequence_descriptor(data);
    const uint64_t num_sequences = 1;
    const auto generators = compute_random_curve25519_generators(data.size(), 10);
    const auto expected_commitment = compute_expected_ristretto255_commitment(data, generators);

    sxt_ristretto255_compressed commitments_data;
    sxt_curve25519_compute_pedersen_commitments_with_generators(
        &commitments_data, num_sequences, &seq_descriptor,
        reinterpret_cast<const sxt_ristretto255*>(generators.data()));
    REQUIRE(*reinterpret_cast<rstt::compressed_element*>(&commitments_data) == expected_commitment);
  }

  cbn::reset_backend_for_testing();
}

//--------------------------------------------------------------------------------------------------
// compute_ristretto255_commitments_with_specified_precomputed_elements
//--------------------------------------------------------------------------------------------------
static void
compute_ristretto255_commitments_with_specified_precomputed_elements(int backend,
                                                                     uint64_t num_precomputed_els) {
  SECTION("We can compute commitments without any provided generators") {
    test_ristretto255_pedersen_commitments_with_given_backend_and_no_generators(
        backend, num_precomputed_els);
  }

  SECTION("We can compute commitments providing generators as input") {
    test_ristretto255_pedersen_commitments_with_given_backend_and_generators(backend,
                                                                             num_precomputed_els);
  }
}

//--------------------------------------------------------------------------------------------------
// compute_ristretto255_commitments_with_given_backend
//--------------------------------------------------------------------------------------------------
static void compute_ristretto255_commitments_with_given_backend(int backend) {
  SECTION("We can compute ristretto255 commitments without precomputing elements") {
    uint64_t num_precomputed_els = 0;
    compute_ristretto255_commitments_with_specified_precomputed_elements(backend,
                                                                         num_precomputed_els);
  }

  SECTION("We can compute ristretto255 commitments using non-zero precomputed elements") {
    uint64_t num_precomputed_els = 10;
    compute_ristretto255_commitments_with_specified_precomputed_elements(backend,
                                                                         num_precomputed_els);
  }
}

TEST_CASE(
    "We can compute pedersen commitments with ristretto255 elements using the naive gpu backend") {
  compute_ristretto255_commitments_with_given_backend(SXT_GPU_BACKEND);
}

TEST_CASE("We can compute pedersen commitments with ristretto255 elements using the pippenger cpu "
          "backend") {
  compute_ristretto255_commitments_with_given_backend(SXT_CPU_BACKEND);
}
