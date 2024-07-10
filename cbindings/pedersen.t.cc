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
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve_bng1/operation/add.h"
#include "sxt/curve_bng1/operation/scalar_multiply.h"
#include "sxt/curve_bng1/random/element_affine.h"
#include "sxt/curve_bng1/type/conversion_utility.h"
#include "sxt/curve_bng1/type/element_affine.h"
#include "sxt/curve_bng1/type/element_p2.h"
#include "sxt/curve_g1/operation/add.h"
#include "sxt/curve_g1/operation/compression.h"
#include "sxt/curve_g1/operation/scalar_multiply.h"
#include "sxt/curve_g1/random/element_affine.h"
#include "sxt/curve_g1/type/compressed_element.h"
#include "sxt/curve_g1/type/conversion_utility.h"
#include "sxt/curve_g1/type/element_affine.h"
#include "sxt/curve_gk/operation/add.h"
#include "sxt/curve_gk/operation/scalar_multiply.h"
#include "sxt/curve_gk/random/element_affine.h"
#include "sxt/curve_gk/type/conversion_utility.h"
#include "sxt/curve_gk/type/element_affine.h"
#include "sxt/curve_gk/type/element_p2.h"
#include "sxt/memory/management/managed_array.h"
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
// get_bls12_381_g1_generators
//--------------------------------------------------------------------------------------------------
static std::vector<cg1t::element_affine> get_bls12_381_g1_generators(uint64_t seq_length,
                                                                     uint64_t offset) {
  std::vector<cg1t::element_affine> generators(seq_length);

  for (uint64_t i = 0; i < seq_length; ++i) {
    basn::fast_random_number_generator rng{static_cast<uint64_t>(i + 1),
                                           static_cast<uint64_t>(i + 2)};
    cg1rn::generate_random_element(generators[i], rng);
  }

  return generators;
}

//--------------------------------------------------------------------------------------------------
// get_bn254_g1_generators
//--------------------------------------------------------------------------------------------------
static std::vector<cn1t::element_affine> get_bn254_g1_generators(uint64_t seq_length,
                                                                 uint64_t offset) {
  std::vector<cn1t::element_affine> generators(seq_length);

  for (uint64_t i = 0; i < seq_length; ++i) {
    basn::fast_random_number_generator rng{static_cast<uint64_t>(i + 1),
                                           static_cast<uint64_t>(i + 2)};
    cn1rn::generate_random_element(generators[i], rng);
  }

  return generators;
}

//--------------------------------------------------------------------------------------------------
// get_grumpkin_generators
//--------------------------------------------------------------------------------------------------
static std::vector<cgkt::element_affine> get_grumpkin_generators(uint64_t seq_length,
                                                                 uint64_t offset) {
  std::vector<cgkt::element_affine> generators(seq_length);

  for (uint64_t i = 0; i < seq_length; ++i) {
    basn::fast_random_number_generator rng{static_cast<uint64_t>(i + 1),
                                           static_cast<uint64_t>(i + 2)};
    cgkrn::generate_random_element(generators[i], rng);
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
// compute_expected_bls12_381_g1_commitment
//--------------------------------------------------------------------------------------------------
template <class T>
static cg1t::compressed_element
compute_expected_bls12_381_g1_commitment(const std::vector<T>& data,
                                         const std::vector<cg1t::element_affine>& generators) {
  SXT_DEBUG_ASSERT(data.size() == generators.size());

  // Convert from affine to projective elements
  memmg::managed_array<cg1t::element_p2> generators_p(generators.size());
  cg1t::batch_to_element_p2(generators_p, generators);

  cg1t::element_p2 expected_commitment{cg1t::element_p2::identity()};

  for (uint64_t i = 0; i < data.size(); ++i) {
    cg1t::element_p2 aux_h{generators_p[i]};
    cg1o::scalar_multiply255(aux_h, aux_h, data[i].data());
    cg1o::add(expected_commitment, expected_commitment, aux_h);
  }

  cg1t::compressed_element expected_commitment_compressed;
  cg1o::compress(expected_commitment_compressed, expected_commitment);

  return expected_commitment_compressed;
}

//--------------------------------------------------------------------------------------------------
// compute_expected_bn254_g1_commitment
//--------------------------------------------------------------------------------------------------
template <class T>
static cn1t::element_affine
compute_expected_bn254_g1_commitment(const std::vector<T>& data,
                                     const std::vector<cn1t::element_affine>& generators) {
  SXT_DEBUG_ASSERT(data.size() == generators.size());

  // Convert from affine to projective elements
  memmg::managed_array<cn1t::element_p2> generators_p(generators.size());
  cn1t::batch_to_element_p2(generators_p, generators);

  cn1t::element_p2 expected_commitment{cn1t::element_p2::identity()};

  for (uint64_t i = 0; i < data.size(); ++i) {
    cn1t::element_p2 aux_h{generators_p[i]};
    cn1o::scalar_multiply255(aux_h, aux_h, data[i].data());
    cn1o::add(expected_commitment, expected_commitment, aux_h);
  }

  cn1t::element_affine expected_commitment_affine;
  cn1t::to_element_affine(expected_commitment_affine, expected_commitment);

  return expected_commitment_affine;
}

//--------------------------------------------------------------------------------------------------
// compute_expected_grumpkin_commitment
//--------------------------------------------------------------------------------------------------
template <class T>
static cgkt::element_affine
compute_expected_grumpkin_commitment(const std::vector<T>& data,
                                     const std::vector<cgkt::element_affine>& generators) {
  SXT_DEBUG_ASSERT(data.size() == generators.size());

  // Convert from affine to projective elements
  memmg::managed_array<cgkt::element_p2> generators_p(generators.size());
  cgkt::batch_to_element_p2(generators_p, generators);

  cgkt::element_p2 expected_commitment{cgkt::element_p2::identity()};

  for (uint64_t i = 0; i < data.size(); ++i) {
    cgkt::element_p2 aux_h{generators_p[i]};
    cgko::scalar_multiply255(aux_h, aux_h, data[i].data());
    cgko::add(expected_commitment, expected_commitment, aux_h);
  }

  cgkt::element_affine expected_commitment_affine;
  cgkt::to_element_affine(expected_commitment_affine, expected_commitment);

  return expected_commitment_affine;
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
    sxt_sequence_descriptor valid_descriptors[] = {
        make_sequence_descriptor(data1),
        make_sequence_descriptor(data2),
    };
    constexpr uint64_t num_sequences = std::size(valid_descriptors);
    rstt::compressed_element commitments_data[num_sequences];
    sxt_curve25519_compute_pedersen_commitments(
        reinterpret_cast<sxt_ristretto255_compressed*>(commitments_data), num_sequences,
        valid_descriptors, 0);
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
    const sxt_sequence_descriptor valid_descriptors[] = {
        make_sequence_descriptor(data_1),
        make_sequence_descriptor(data_2),
        make_sequence_descriptor(data_3),
    };
    constexpr uint64_t num_sequences = std::size(valid_descriptors);

    // we verify that `c = scal * a + b` implies that `commit_c = scal * commit_a + commit_b`
    rstt::compressed_element commitments_data[num_sequences];
    sxt_curve25519_compute_pedersen_commitments(
        reinterpret_cast<sxt_ristretto255_compressed*>(commitments_data), num_sequences,
        valid_descriptors, offset_gens);

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
// test_bls12_381_g1_pedersen_commitments_with_given_backend_and_generators
//--------------------------------------------------------------------------------------------------
static void test_bls12_381_g1_pedersen_commitments_with_given_backend_and_generators(
    int backend, uint64_t num_precomputed_generators) {
  initialize_backend(backend, num_precomputed_generators);

  SECTION("We verify that using the correct generators will produce correct results") {
    constexpr std::array<uint8_t, 32> a{0x1b, 0xa7, 0x6d, 0xa5, 0x98, 0x82, 0x56, 0x2b,
                                        0xd2, 0x19, 0xf5, 0xe,  0xc8, 0xfa, 0x5,  0x85,
                                        0x91, 0xe7, 0x1d, 0x5e, 0xd2, 0x60, 0x22, 0x10,
                                        0x6a, 0xdc, 0x18, 0xfd, 0xfc, 0xf8, 0x9a, 0xc};
    constexpr std::array<uint8_t, 32> b{0x01, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

    const std::vector<std::array<uint8_t, 32>> data = {a, b};
    const auto seq_descriptor = make_sequence_descriptor(data);
    constexpr uint64_t num_sequences{1};
    const auto generators = get_bls12_381_g1_generators(data.size(), 10);
    const auto expected_commitment = compute_expected_bls12_381_g1_commitment(data, generators);

    sxt_bls12_381_g1_compressed commitments_data;
    sxt_bls12_381_g1_compute_pedersen_commitments_with_generators(
        &commitments_data, num_sequences, &seq_descriptor,
        reinterpret_cast<const sxt_bls12_381_g1*>(generators.data()));
    REQUIRE(*reinterpret_cast<cg1t::compressed_element*>(&commitments_data) == expected_commitment);
  }

  cbn::reset_backend_for_testing();
}

//--------------------------------------------------------------------------------------------------
// test_bn254_g1_pedersen_commitments_with_given_backend_and_generators
//--------------------------------------------------------------------------------------------------
static void test_bn254_g1_pedersen_commitments_with_given_backend_and_generators(
    int backend, uint64_t num_precomputed_generators) {
  initialize_backend(backend, num_precomputed_generators);

  SECTION("We verify that using the correct generators will produce correct results") {
    constexpr std::array<uint8_t, 32> a{0x1b, 0xa7, 0x6d, 0xa5, 0x98, 0x82, 0x56, 0x2b,
                                        0xd2, 0x19, 0xf5, 0xe,  0xc8, 0xfa, 0x5,  0x85,
                                        0x91, 0xe7, 0x1d, 0x5e, 0xd2, 0x60, 0x22, 0x10,
                                        0x6a, 0xdc, 0x18, 0xfd, 0xfc, 0xf8, 0x9a, 0xc};
    constexpr std::array<uint8_t, 32> b{0x01, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

    const std::vector<std::array<uint8_t, 32>> data = {a, b};
    const auto seq_descriptor = make_sequence_descriptor(data);
    constexpr uint64_t num_sequences{1};
    const auto generators = get_bn254_g1_generators(data.size(), 10);
    const auto expected_commitment = compute_expected_bn254_g1_commitment(data, generators);

    sxt_bn254_g1 commitments_data;
    sxt_bn254_g1_uncompressed_compute_pedersen_commitments_with_generators(
        &commitments_data, num_sequences, &seq_descriptor,
        reinterpret_cast<const sxt_bn254_g1*>(generators.data()));
    REQUIRE(*reinterpret_cast<cn1t::element_affine*>(&commitments_data) == expected_commitment);
  }

  cbn::reset_backend_for_testing();
}

//--------------------------------------------------------------------------------------------------
// test_grumpkin_pedersen_commitments_with_given_backend_and_generators
//--------------------------------------------------------------------------------------------------
static void test_grumpkin_pedersen_commitments_with_given_backend_and_generators(
    int backend, uint64_t num_precomputed_generators) {
  initialize_backend(backend, num_precomputed_generators);

  SECTION("We verify that using the correct generators will produce correct results") {
    constexpr std::array<uint8_t, 32> a{0x1b, 0xa7, 0x6d, 0xa5, 0x98, 0x82, 0x56, 0x2b,
                                        0xd2, 0x19, 0xf5, 0xe,  0xc8, 0xfa, 0x5,  0x85,
                                        0x91, 0xe7, 0x1d, 0x5e, 0xd2, 0x60, 0x22, 0x10,
                                        0x6a, 0xdc, 0x18, 0xfd, 0xfc, 0xf8, 0x9a, 0xc};
    constexpr std::array<uint8_t, 32> b{0x01, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

    const std::vector<std::array<uint8_t, 32>> data = {a, b};
    const auto seq_descriptor = make_sequence_descriptor(data);
    constexpr uint64_t num_sequences{1};
    const auto generators = get_grumpkin_generators(data.size(), 10);
    const auto expected_commitment = compute_expected_grumpkin_commitment(data, generators);

    sxt_grumpkin commitments_data;
    sxt_grumpkin_uncompressed_compute_pedersen_commitments_with_generators(
        &commitments_data, num_sequences, &seq_descriptor,
        reinterpret_cast<const sxt_grumpkin*>(generators.data()));
    REQUIRE(*reinterpret_cast<cgkt::element_affine*>(&commitments_data) == expected_commitment);
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
// compute_bls12_381_g1_commitments_with_specified_precomputed_elements
//--------------------------------------------------------------------------------------------------
static void
compute_bls12_381_g1_commitments_with_specified_precomputed_elements(int backend,
                                                                     uint64_t num_precomputed_els) {
  SECTION("We can compute commitments providing generators as input") {
    test_bls12_381_g1_pedersen_commitments_with_given_backend_and_generators(backend,
                                                                             num_precomputed_els);
  }
}

//--------------------------------------------------------------------------------------------------
// compute_bn254_g1_commitments_with_specified_precomputed_elements
//--------------------------------------------------------------------------------------------------
static void
compute_bn254_g1_commitments_with_specified_precomputed_elements(int backend,
                                                                 uint64_t num_precomputed_els) {
  SECTION("We can compute commitments providing generators as input") {
    test_bn254_g1_pedersen_commitments_with_given_backend_and_generators(backend,
                                                                         num_precomputed_els);
  }
}

//--------------------------------------------------------------------------------------------------
// compute_grumpkin_commitments_with_specified_precomputed_elements
//--------------------------------------------------------------------------------------------------
static void
compute_grumpkin_commitments_with_specified_precomputed_elements(int backend,
                                                                 uint64_t num_precomputed_els) {
  SECTION("We can compute commitments providing generators as input") {
    test_grumpkin_pedersen_commitments_with_given_backend_and_generators(backend,
                                                                         num_precomputed_els);
  }
}

//--------------------------------------------------------------------------------------------------
// compute_ristretto255_commitments_with_given_backend
//--------------------------------------------------------------------------------------------------
static void compute_ristretto255_commitments_with_given_backend(int backend) {
  SECTION("We can compute commitments without precomputing elements") {
    uint64_t num_precomputed_els = 0;
    compute_ristretto255_commitments_with_specified_precomputed_elements(backend,
                                                                         num_precomputed_els);
  }

  SECTION("We can compute commitments using non-zero precomputed elements") {
    uint64_t num_precomputed_els{10};
    compute_ristretto255_commitments_with_specified_precomputed_elements(backend,
                                                                         num_precomputed_els);
  }
}

//--------------------------------------------------------------------------------------------------
// compute_bls12_381_g1_commitments_with_given_backend
//--------------------------------------------------------------------------------------------------
static void compute_bls12_381_g1_commitments_with_given_backend(int backend) {
  SECTION("We can compute commitments without precomputing elements") {
    uint64_t num_precomputed_els{0};
    compute_bls12_381_g1_commitments_with_specified_precomputed_elements(backend,
                                                                         num_precomputed_els);
  }

  SECTION("We can compute commitments using non-zero precomputed elements") {
    uint64_t num_precomputed_els{10};
    compute_bls12_381_g1_commitments_with_specified_precomputed_elements(backend,
                                                                         num_precomputed_els);
  }
}

//--------------------------------------------------------------------------------------------------
// compute_bn254_g1_commitments_with_given_backend
//--------------------------------------------------------------------------------------------------
static void compute_bn254_g1_commitments_with_given_backend(int backend) {
  SECTION("We can compute commitments without precomputing elements") {
    uint64_t num_precomputed_els{0};
    compute_bn254_g1_commitments_with_specified_precomputed_elements(backend, num_precomputed_els);
  }

  SECTION("We can compute commitments using non-zero precomputed elements") {
    uint64_t num_precomputed_els{10};
    compute_bn254_g1_commitments_with_specified_precomputed_elements(backend, num_precomputed_els);
  }
}

//--------------------------------------------------------------------------------------------------
// compute_grumpkin_commitments_with_given_backend
//--------------------------------------------------------------------------------------------------
static void compute_grumpkin_commitments_with_given_backend(int backend) {
  SECTION("We can compute commitments without precomputing elements") {
    uint64_t num_precomputed_els{0};
    compute_grumpkin_commitments_with_specified_precomputed_elements(backend, num_precomputed_els);
  }

  SECTION("We can compute commitments using non-zero precomputed elements") {
    uint64_t num_precomputed_els{10};
    compute_grumpkin_commitments_with_specified_precomputed_elements(backend, num_precomputed_els);
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

TEST_CASE(
    "We can compute pedersen commitments with bls12-381 G1 elements using the naive gpu backend") {
  compute_bls12_381_g1_commitments_with_given_backend(SXT_GPU_BACKEND);
}

TEST_CASE("We can compute pedersen commitments with bls12-381 G1 elements using the pippenger cpu "
          "backend") {
  compute_bls12_381_g1_commitments_with_given_backend(SXT_CPU_BACKEND);
}

TEST_CASE(
    "We can compute pedersen commitments with bn254 G1 elements using the naive gpu backend") {
  compute_bn254_g1_commitments_with_given_backend(SXT_GPU_BACKEND);
}

TEST_CASE("We can compute pedersen commitments with bn254 G1 elements using the pippenger cpu "
          "backend") {
  compute_bn254_g1_commitments_with_given_backend(SXT_CPU_BACKEND);
}

TEST_CASE(
    "We can compute pedersen commitments with Grumpkin elements using the naive gpu backend") {
  compute_grumpkin_commitments_with_given_backend(SXT_GPU_BACKEND);
}

TEST_CASE("We can compute pedersen commitments with Grumpkin elements using the pippenger cpu "
          "backend") {
  compute_grumpkin_commitments_with_given_backend(SXT_CPU_BACKEND);
}
