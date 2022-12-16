#include "sxt/seqcommit/cbindings/pedersen.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/operation/add.h"
#include "sxt/ristretto/operation/scalar_multiply.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/cbindings/backend.h"
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
// make_sequence_descriptor
//--------------------------------------------------------------------------------------------------
template <class T>
static sxt_sequence_descriptor make_sequence_descriptor(const std::vector<T>& data,
                                                        const std::vector<uint64_t>& indices) {
  if (indices.size() > 0) {
    assert(data.size() == indices.size());
  }

  return {.element_nbytes = sizeof(T),
          .n = data.size(),
          .data = reinterpret_cast<const uint8_t*>(data.data()),
          .indices = indices.data()};
}

//--------------------------------------------------------------------------------------------------
// test_pedersen_commitments_with_given_backend_and_no_generators
//--------------------------------------------------------------------------------------------------
static void test_pedersen_commitments_with_given_backend_and_no_generators(
    int backend, uint64_t num_precomputed_generators) {
  initialize_backend(backend, num_precomputed_generators);

  SECTION("Null input commitment pointer will error out") {
    const std::vector<uint8_t> data(1);
    const std::vector<uint64_t> indices(0);
    sxt_compressed_ristretto* commitment = nullptr;
    const auto seq_descriptor = make_sequence_descriptor(data, indices);
    const uint32_t num_sequences = 1;
    REQUIRE(sxt_compute_pedersen_commitments(commitment, num_sequences, &seq_descriptor) != 0);
  }

  SECTION("Null input sequence pointer and num_sequence bigger than zero will error out") {
    const uint32_t num_sequences = 1;
    const sxt_sequence_descriptor* invalid_descriptor = nullptr;
    sxt_compressed_ristretto commitment;
    REQUIRE(sxt_compute_pedersen_commitments(&commitment, num_sequences, invalid_descriptor) != 0);
  }

  SECTION("Zero num_sequence will not error out even with a null input sequence pointer") {
    const uint32_t num_sequences = 0;
    const sxt_sequence_descriptor* invalid_descriptor = nullptr;
    sxt_compressed_ristretto commitment{1u};
    REQUIRE(sxt_compute_pedersen_commitments(&commitment, num_sequences, invalid_descriptor) == 0);
    REQUIRE(rstt::compressed_element{1u} ==
            *reinterpret_cast<rstt::compressed_element*>(&commitment));
  }

  SECTION("Input sequences with zero length will not error out even with a null data pointer") {
    const std::vector<uint8_t> data(0);
    const std::vector<uint64_t> indices(0);
    const auto seq_descriptor = make_sequence_descriptor(data, indices);
    const uint32_t num_sequences = 1;
    sxt_compressed_ristretto commitment;
    REQUIRE(sxt_compute_pedersen_commitments(&commitment, num_sequences, &seq_descriptor) == 0);
    REQUIRE(rstt::compressed_element() ==
            *reinterpret_cast<rstt::compressed_element*>(&commitment));
  }

  SECTION("Null data pointer with a non-zero sequence length will error out") {
    const uint64_t sequence_length = 1;
    const sxt_sequence_descriptor invalid_descriptor = {.element_nbytes = sizeof(uint8_t),
                                                        .n = sequence_length,
                                                        .data = nullptr,
                                                        .indices = nullptr};
    const uint32_t num_sequences = 1;
    sxt_compressed_ristretto commitment;
    REQUIRE(sxt_compute_pedersen_commitments(&commitment, num_sequences, &invalid_descriptor) != 0);
  }

  SECTION("Out of range sequence element_nbytes will error out") {
    SECTION("when element_nbytes is smaller than 1") {
      const std::vector<uint8_t> data(1);
      const uint64_t sequence_length = data.size();
      const std::vector<sxt_sequence_descriptor> invalid_descriptor = {
          {.element_nbytes = 0, .n = sequence_length, .data = data.data(), .indices = nullptr}};
      const uint32_t num_sequences = 1;
      sxt_compressed_ristretto commitment;
      REQUIRE(sxt_compute_pedersen_commitments(&commitment, num_sequences,
                                               invalid_descriptor.data()) != 0);
    }

    SECTION("when element_nbytes is bigger than 32") {
      const std::vector<uint8_t> data(34);
      const uint64_t sequence_length = 1;
      const std::vector<sxt_sequence_descriptor> invalid_descriptor = {
          {.element_nbytes = 33, .n = sequence_length, .data = data.data(), .indices = nullptr}};
      const uint32_t num_sequences = invalid_descriptor.size();
      sxt_compressed_ristretto commitment;
      REQUIRE(sxt_compute_pedersen_commitments(&commitment, num_sequences,
                                               invalid_descriptor.data()) != 0);
    }
  }

  SECTION("We can multiply and add two commitments together, then compare them against the c "
          "binding results") {
    const uint64_t scal = 52;
    const std::vector<uint64_t> indices = {};
    const std::vector<uint64_t> data_1 = {2000, 7500, 5000, 1500};
    const std::vector<uint64_t> data_2 = {5000, 0, 400000, 10};
    const std::vector<uint64_t> data_3 = {
        scal * data_1[0] + data_2[0], scal * data_1[1] + data_2[1], scal * data_1[2] + data_2[2],
        scal * data_1[3] + data_2[3]};
    const std::vector<sxt_sequence_descriptor> valid_descriptors = {
        make_sequence_descriptor(data_1, indices),
        make_sequence_descriptor(data_2, indices),
        make_sequence_descriptor(data_3, indices),
    };
    const uint64_t num_sequences = valid_descriptors.size();

    // we verify that `c = scal * a + b` implies that `commit_c = scal * commit_a + commit_b`
    rstt::compressed_element commitments_data[num_sequences];
    REQUIRE(sxt_compute_pedersen_commitments(
                reinterpret_cast<sxt_compressed_ristretto*>(commitments_data), num_sequences,
                valid_descriptors.data()) == 0);

    auto commit_a = commitments_data[0], commit_b = commitments_data[1],
         commit_c = commitments_data[0];
    rsto::scalar_multiply(commit_a, scal, commit_a);
    rsto::add(commit_c, commit_a, commit_b);

    REQUIRE(rstt::compressed_element() != commit_c);
    REQUIRE(commit_c == commitments_data[2]);
  }

  SECTION("We can correctly compute sparse commitments") {
    const std::vector<uint8_t> sparse_data = {1, 2, 3, 4, 9};
    const std::vector<uint8_t> dense_data = {1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0};
    const std::vector<uint64_t> dense_indices = {}, sparse_indices = {0, 2, 4, 5, 9};
    const auto dense_descriptor = make_sequence_descriptor(dense_data, dense_indices);
    const auto sparse_descriptor = make_sequence_descriptor(sparse_data, sparse_indices);
    const std::vector<sxt_sequence_descriptor> descriptors = {dense_descriptor, sparse_descriptor};
    const uint64_t num_sequences = descriptors.size();

    // we verify that both sparse and dense results are equal
    rstt::compressed_element commitments_data[num_sequences];
    REQUIRE(sxt_compute_pedersen_commitments(
                reinterpret_cast<sxt_compressed_ristretto*>(commitments_data), num_sequences,
                descriptors.data()) == 0);
    REQUIRE(rstt::compressed_element() != commitments_data[0]);
    REQUIRE(commitments_data[0] == commitments_data[1]);
  }

  SECTION("We can correctly compute dense commitments as sparse commitments") {
    const std::vector<uint8_t> dense_data = {1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0};
    const std::vector<uint8_t> sparse_data = {1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0};
    const std::vector<uint64_t> dense_indices = {},
                                sparse_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const auto dense_descriptor = make_sequence_descriptor(dense_data, dense_indices);
    const auto sparse_descriptor = make_sequence_descriptor(sparse_data, sparse_indices);
    const std::vector<sxt_sequence_descriptor> descriptors = {dense_descriptor, sparse_descriptor};
    const uint64_t num_sequences = descriptors.size();

    // dense_commitment result is the same as the sparse commitment
    rstt::compressed_element commitments_data[num_sequences];
    REQUIRE(sxt_compute_pedersen_commitments(
                reinterpret_cast<sxt_compressed_ristretto*>(commitments_data), num_sequences,
                descriptors.data()) == 0);

    REQUIRE(rstt::compressed_element() != commitments_data[0]);
    REQUIRE(commitments_data[0] == commitments_data[1]);
  }

  sqccb::reset_backend_for_testing();
}

//--------------------------------------------------------------------------------------------------
// compute_random_generators
//--------------------------------------------------------------------------------------------------
static std::vector<c21t::element_p3> compute_random_generators(uint64_t seq_length) {
  std::vector<c21t::element_p3> generators(seq_length);

  for (uint64_t i = 0; i < seq_length; ++i) {
    sqcgn::compute_base_element(generators[i], 1 + i * i);
  }

  return generators;
}

//--------------------------------------------------------------------------------------------------
// compute_expected_commitment
//--------------------------------------------------------------------------------------------------
template <class T>
static rstt::compressed_element
compute_expected_commitment(const std::vector<T>& data,
                            const std::vector<c21t::element_p3>& generators) {
  assert(data.size() == generators.size());

  rstt::compressed_element expected_commitment;
  rstb::to_bytes(expected_commitment.data(), c21cn::zero_p3_v);

  for (uint64_t i = 0; i < data.size(); ++i) {
    rstt::compressed_element aux_h;
    rstb::to_bytes(aux_h.data(), generators[i]);
    rsto::scalar_multiply(aux_h, data[i], aux_h);
    rsto::add(expected_commitment, expected_commitment, aux_h);
  }

  return expected_commitment;
}

//--------------------------------------------------------------------------------------------------
// test_pedersen_commitments_with_given_backend_and_generators
//--------------------------------------------------------------------------------------------------
static void
test_pedersen_commitments_with_given_backend_and_generators(int backend,
                                                            uint64_t num_precomputed_generators) {
  initialize_backend(backend, num_precomputed_generators);

  SECTION("We verify that using null generator pointers will error out") {
    const std::vector<uint64_t> indices = {};
    const std::vector<uint32_t> data = {2000, 7500};
    const std::vector<sxt_ristretto> generators_data = {};
    const auto seq_descriptor = make_sequence_descriptor(data, indices);
    const uint64_t num_sequences = 1;

    sxt_compressed_ristretto commitments_data;
    REQUIRE(sxt_compute_pedersen_commitments_with_generators(
                &commitments_data, num_sequences, &seq_descriptor, generators_data.data()) != 0);
  }

  SECTION("We verify that using the correct generators will produce correct results") {
    const std::vector<uint64_t> indices = {};
    const std::vector<uint32_t> data = {2000, 7500};
    const auto seq_descriptor = make_sequence_descriptor(data, indices);
    const uint64_t num_sequences = 1;
    const auto generators = compute_random_generators(data.size());
    const auto expected_commitment = compute_expected_commitment(data, generators);

    sxt_compressed_ristretto commitments_data;
    REQUIRE(sxt_compute_pedersen_commitments_with_generators(
                &commitments_data, num_sequences, &seq_descriptor,
                reinterpret_cast<const sxt_ristretto*>(generators.data())) == 0);
    REQUIRE(*reinterpret_cast<rstt::compressed_element*>(&commitments_data) == expected_commitment);
  }

  SECTION("We verify that sparse sequence indices are ignored when generators are provided") {
    const std::vector<uint8_t> sparse_data = {1, 3};
    const std::vector<uint64_t> sparse_indices = {0, 5};
    const auto generators = compute_random_generators(sparse_data.size());
    const auto expected_commitment = compute_expected_commitment(sparse_data, generators);
    const auto sparse_descriptor = make_sequence_descriptor(sparse_data, sparse_indices);
    const uint64_t num_sequences = 1;

    sxt_compressed_ristretto commitments_data;
    REQUIRE(sxt_compute_pedersen_commitments_with_generators(
                &commitments_data, num_sequences, &sparse_descriptor,
                reinterpret_cast<const sxt_ristretto*>(generators.data())) == 0);
    REQUIRE(*reinterpret_cast<rstt::compressed_element*>(&commitments_data) == expected_commitment);
  }

  sqccb::reset_backend_for_testing();
}

//--------------------------------------------------------------------------------------------------
// compute_commitments_with_specified_precomputed_elements
//--------------------------------------------------------------------------------------------------
static void compute_commitments_with_specified_precomputed_elements(int backend,
                                                                    uint64_t num_precomputed_els) {
  SECTION("We can compute commitments without any provided generators") {
    test_pedersen_commitments_with_given_backend_and_no_generators(backend, num_precomputed_els);
  }

  SECTION("We can compute commitments providing generators as input") {
    test_pedersen_commitments_with_given_backend_and_generators(backend, num_precomputed_els);
  }
}

//--------------------------------------------------------------------------------------------------
// compute_commitments_with_given_backend
//--------------------------------------------------------------------------------------------------
static void compute_commitments_with_given_backend(int backend) {
  SECTION("We can compute commitments without precomputing elements") {
    uint64_t num_precomputed_els = 0;
    compute_commitments_with_specified_precomputed_elements(backend, num_precomputed_els);
  }

  SECTION("We can compute commitments using non-zero precomputed elements") {
    uint64_t num_precomputed_els = 10;
    compute_commitments_with_specified_precomputed_elements(backend, num_precomputed_els);
  }
}

TEST_CASE("We can compute pedersen commitments using the naive cpu backend") {
  compute_commitments_with_given_backend(SXT_NAIVE_BACKEND_CPU);
}

TEST_CASE("We can compute pedersen commitments using the naive gpu backend") {
  compute_commitments_with_given_backend(SXT_NAIVE_BACKEND_GPU);
}

TEST_CASE("We can compute pedersen commitments using the pippenger cpu backend") {
  compute_commitments_with_given_backend(SXT_PIPPENGER_BACKEND_CPU);
}
