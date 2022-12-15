#include "sxt/seqcommit/cbindings/pedersen.h"

#include <algorithm>
#include <array>
#include <string>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/operation/add.h"
#include "sxt/ristretto/operation/scalar_multiply.h"
#include "sxt/ristretto/random/element.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/cbindings/backend.h"
#include "sxt/seqcommit/generator/base_element.h"

using namespace sxt;
using namespace sxt::sqccb;

TEST_CASE("run pedersen initialization and finalize tests") {
  sxt_compressed_ristretto commitments[1];
  sxt_sequence_descriptor valid_descriptors[1];

  SECTION("not initialized library will error out") {
    int ret = sxt_compute_pedersen_commitments(commitments, 0, valid_descriptors);

    REQUIRE(ret != 0);
  }

  SECTION("correct naive cpu backend will not error out") {
    const sxt_config config = {SXT_NAIVE_BACKEND_CPU};
    REQUIRE(sxt_init(&config) == 0);

    int ret = sxt_compute_pedersen_commitments(commitments, 0, valid_descriptors);

    REQUIRE(ret == 0);

    sqccb::reset_backend_for_testing();
  }

  SECTION("correct naive gpu backend will not error out") {
    const sxt_config config = {SXT_NAIVE_BACKEND_GPU};
    REQUIRE(sxt_init(&config) == 0);

    int ret = sxt_compute_pedersen_commitments(commitments, 0, valid_descriptors);

    REQUIRE(ret == 0);

    sqccb::reset_backend_for_testing();
  }

  SECTION("correct pippenger cpu backend will not error out") {
    const sxt_config config = {SXT_PIPPENGER_BACKEND_CPU};
    REQUIRE(sxt_init(&config) == 0);

    int ret = sxt_compute_pedersen_commitments(commitments, 0, valid_descriptors);

    REQUIRE(ret == 0);

    sqccb::reset_backend_for_testing();
  }
}

static void test_pedersen_commitments_with_given_backend(int backend,
                                                         uint64_t num_precomputed_generators,
                                                         std::string backend_name) {
  ////////////////////////////////////////////////////////////////
  // sxt_compute_pedersen_commitments
  ////////////////////////////////////////////////////////////////
  SECTION(backend_name + " - We can compute commitments without providing any generator") {

    SECTION("correct initialization and input will not error out") {
      const sxt_config config = {backend, num_precomputed_generators};
      REQUIRE(sxt_init(&config) == 0);

      uint8_t data[33];
      sxt_compressed_ristretto commitment;
      sxt_sequence_descriptor valid_seq_descriptor = {.element_nbytes = 1, // number bytes
                                                      .n = 1,              // number rows
                                                      .data = data,        // data pointer
                                                      .indices = nullptr};

      int ret = sxt_compute_pedersen_commitments(&commitment, 1, &valid_seq_descriptor);

      REQUIRE(ret == 0);

      sqccb::reset_backend_for_testing();
    }

    SECTION("null commitment pointers will error out") {
      const sxt_config config = {backend, num_precomputed_generators};
      REQUIRE(sxt_init(&config) == 0);

      uint8_t data[33];
      sxt_sequence_descriptor valid_descriptor = {.element_nbytes = 1, // number bytes
                                                  .n = 1,              // number rows
                                                  .data = data,        // data pointer
                                                  nullptr};

      int ret = sxt_compute_pedersen_commitments(nullptr, 1, &valid_descriptor);

      REQUIRE(ret != 0);

      sqccb::reset_backend_for_testing();
    }

    SECTION("null value_sequences will error out") {
      const sxt_config config = {backend, num_precomputed_generators};
      REQUIRE(sxt_init(&config) == 0);

      sxt_compressed_ristretto commitment;
      int ret = sxt_compute_pedersen_commitments(&commitment, 1, nullptr);

      REQUIRE(ret != 0);

      sqccb::reset_backend_for_testing();
    }

    SECTION("zero sequences will not error out") {
      const sxt_config config = {backend, num_precomputed_generators};
      REQUIRE(sxt_init(&config) == 0);

      sxt_compressed_ristretto commitment;
      int ret = sxt_compute_pedersen_commitments(&commitment, 0, nullptr);

      REQUIRE(ret == 0);

      sqccb::reset_backend_for_testing();
    }

    SECTION("zero length commitments will not error out") {
      const sxt_config config = {backend, num_precomputed_generators};
      REQUIRE(sxt_init(&config) == 0);

      uint8_t data[33];
      sxt_compressed_ristretto commitment;
      sxt_sequence_descriptor zero_length_seq_descriptor = {.element_nbytes = 1, // number bytes
                                                            .n = 0,              // number rows
                                                            .data = data,        // data pointer
                                                            nullptr};

      int ret = sxt_compute_pedersen_commitments(&commitment, 1, &zero_length_seq_descriptor);

      REQUIRE(ret == 0);

      REQUIRE(rstt::compressed_element() ==
              reinterpret_cast<rstt::compressed_element*>(&commitment)[0]);

      sqccb::reset_backend_for_testing();
    }

    SECTION("out of range (< 1 or > 32) element_nbytes will error out") {
      const sxt_config config = {backend, num_precomputed_generators};
      REQUIRE(sxt_init(&config) == 0);

      SECTION("element_nbytes == 0 (< 1) error out") {
        uint8_t data[33];
        sxt_compressed_ristretto commitment;
        sxt_sequence_descriptor invalid_descriptors = {.element_nbytes = 0, // number bytes
                                                       .n = 1,              // number rows
                                                       .data = data,        // data pointer
                                                       .indices = nullptr};

        int ret = sxt_compute_pedersen_commitments(&commitment, 1, &invalid_descriptors);

        REQUIRE(ret != 0);
      }

      SECTION("element_nbytes == 33 (> 32) error out") {
        uint8_t data[33];
        sxt_compressed_ristretto commitment;
        sxt_sequence_descriptor invalid_descriptors = {.element_nbytes = 33, // number bytes
                                                       .n = 1,               // number rows
                                                       .data = data,         // data pointer
                                                       .indices = nullptr};

        int ret = sxt_compute_pedersen_commitments(&commitment, 1, &invalid_descriptors);

        REQUIRE(ret != 0);
      }

      sqccb::reset_backend_for_testing();
    }

    SECTION("null element data pointer will error out") {
      const sxt_config config = {backend, num_precomputed_generators};
      REQUIRE(sxt_init(&config) == 0);

      sxt_compressed_ristretto commitment;
      sxt_sequence_descriptor invalid_descriptors = {.element_nbytes = 1, // number bytes
                                                     .n = 1,              // number rows
                                                     .data = nullptr,     // null data pointer
                                                     .indices = nullptr};

      int ret = sxt_compute_pedersen_commitments(&commitment, 1, &invalid_descriptors);

      REQUIRE(ret != 0);

      sqccb::reset_backend_for_testing();
    }

    SECTION("We can multiply and add two commitments together") {
      const sxt_config config = {backend, num_precomputed_generators};
      REQUIRE(sxt_init(&config) == 0);

      const uint64_t num_rows = 4;
      const uint64_t num_sequences = 3;
      const uint8_t element_nbytes = sizeof(int);
      const unsigned int multiplicative_constant = 52;

      sxt_compressed_ristretto commitments_data[num_sequences];

      const int query[num_sequences][num_rows] = {
          {2000, 7500, 5000, 1500},
          {5000, 0, 400000, 10},
          {multiplicative_constant * 2000 + 5000, multiplicative_constant * 7500 + 0,
           multiplicative_constant * 5000 + 400000, multiplicative_constant * 1500 + 10}};

      sxt_sequence_descriptor valid_descriptors[num_sequences];

      // populating sequence object
      for (uint64_t i = 0; i < num_sequences; ++i) {
        sxt_sequence_descriptor descriptor = {element_nbytes, num_rows,
                                              reinterpret_cast<const uint8_t*>(query[i]), nullptr};

        valid_descriptors[i] = descriptor;
      }

      SECTION("c = 52 * a + b ==> commit_c = 52 * commit_a + commit_b") {
        int ret =
            sxt_compute_pedersen_commitments(commitments_data, num_sequences, valid_descriptors);

        REQUIRE(ret == 0);

        rstt::compressed_element p =
            reinterpret_cast<rstt::compressed_element*>(commitments_data)[0];

        rstt::compressed_element q =
            reinterpret_cast<rstt::compressed_element*>(commitments_data)[1];
        ;

        rsto::scalar_multiply(p, multiplicative_constant, p); // h_i = a_i * g_i

        rsto::add(p, p, q);

        rstt::compressed_element expected_commitment_c =
            reinterpret_cast<rstt::compressed_element*>(commitments_data)[2];

        // verify that result is not null
        REQUIRE(rstt::compressed_element() != p);

        REQUIRE(p == expected_commitment_c);
      }

      sqccb::reset_backend_for_testing();
    }

    SECTION("We can compute sparse commitments") {
      const sxt_config config = {backend, num_precomputed_generators};
      REQUIRE(sxt_init(&config) == 0);

      const uint64_t num_sequences = 2;
      const uint8_t element_nbytes = sizeof(int);

      sxt_compressed_ristretto commitments_data[num_sequences];

      const int dense_data[11] = {1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0};

      const int sparse_data[5] = {1, 2, 3, 4, 9};

      const uint64_t sparse_indices[5] = {0, 2, 4, 5, 9};

      sxt_sequence_descriptor dense_descriptor = {
          element_nbytes, 11, reinterpret_cast<const uint8_t*>(dense_data), nullptr};

      sxt_sequence_descriptor sparse_descriptor = {
          element_nbytes, 5, reinterpret_cast<const uint8_t*>(sparse_data), sparse_indices};

      sxt_sequence_descriptor descriptors[num_sequences] = {dense_descriptor, sparse_descriptor};

      SECTION("sparse_commitments == dense_commitment") {
        int ret = sxt_compute_pedersen_commitments(commitments_data, num_sequences, descriptors);

        REQUIRE(ret == 0);

        rstt::compressed_element& dense_commitment =
            reinterpret_cast<rstt::compressed_element*>(commitments_data)[0];

        rstt::compressed_element& sparse_commitment =
            reinterpret_cast<rstt::compressed_element*>(commitments_data)[1];

        // verify that result is not null
        REQUIRE(rstt::compressed_element() != dense_commitment);

        REQUIRE(dense_commitment == sparse_commitment);
      }

      sqccb::reset_backend_for_testing();
    }

    SECTION("we can compute dense commitments as sparse commitments") {
      const sxt_config config = {backend, num_precomputed_generators};
      REQUIRE(sxt_init(&config) == 0);

      const uint64_t num_sequences = 2;
      const uint8_t element_nbytes = sizeof(int);

      sxt_compressed_ristretto commitments_data[num_sequences];

      const int dense_data[11] = {1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0};

      const int sparse_data[11] = {1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0};

      const uint64_t sparse_indices[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

      sxt_sequence_descriptor dense_descriptor = {
          element_nbytes, 11, reinterpret_cast<const uint8_t*>(dense_data), nullptr};

      sxt_sequence_descriptor sparse_descriptor = {
          element_nbytes, 11, reinterpret_cast<const uint8_t*>(sparse_data), sparse_indices};

      sxt_sequence_descriptor descriptors[num_sequences] = {dense_descriptor, sparse_descriptor};

      SECTION("dense_commitment == dense_commitment as sparse") {
        int ret = sxt_compute_pedersen_commitments(commitments_data, num_sequences, descriptors);

        REQUIRE(ret == 0);

        rstt::compressed_element& dense_commitment =
            reinterpret_cast<rstt::compressed_element*>(commitments_data)[0];

        rstt::compressed_element& sparse_commitment =
            reinterpret_cast<rstt::compressed_element*>(commitments_data)[1];

        // verify that result is not null
        REQUIRE(rstt::compressed_element() != dense_commitment);

        REQUIRE(dense_commitment == sparse_commitment);
      }

      sqccb::reset_backend_for_testing();
    }
  }

  ////////////////////////////////////////////////////////////////
  // sxt_compute_pedersen_commitments_with_generators
  ////////////////////////////////////////////////////////////////

  SECTION(backend_name + " - We can compute commitments with provided generators") {
    const sxt_config config = {backend, num_precomputed_generators};
    REQUIRE(sxt_init(&config) == 0);

    const uint64_t num_rows = 4;
    const uint64_t num_sequences = 1;
    const uint8_t element_nbytes = sizeof(int);

    sxt_ristretto generators_data[num_rows];
    sxt_compressed_ristretto commitments_data[num_sequences];

    const uint32_t query[num_rows] = {2000, 7500, 5000, 1500};
    rstt::compressed_element expected_g;

    for (uint64_t i = 0; i < num_rows; ++i) {
      c21t::element_p3& g_i = reinterpret_cast<c21t::element_p3*>(generators_data)[i];

      sqcgn::compute_base_element(g_i, query[i]);

      rstt::compressed_element h;
      rstb::to_bytes(h.data(), g_i);

      rsto::scalar_multiply(h, query[i], h);

      rsto::add(expected_g, expected_g, h);
    }

    rstt::compressed_element expected_commitment_c = expected_g;

    sxt_sequence_descriptor valid_descriptors[num_sequences];

    // populating sequence object
    for (uint64_t i = 0; i < num_sequences; ++i) {
      sxt_sequence_descriptor descriptor = {element_nbytes, num_rows,
                                            reinterpret_cast<const uint8_t*>(query), nullptr};

      valid_descriptors[i] = descriptor;
    }

    SECTION("passing null generators will error out") {
      int ret = sxt_compute_pedersen_commitments_with_generators(commitments_data, num_sequences,
                                                                 valid_descriptors, nullptr);

      REQUIRE(ret != 0);
    }

    SECTION("passing valid generators will not error out") {
      int ret = sxt_compute_pedersen_commitments_with_generators(
          commitments_data, num_sequences, valid_descriptors, generators_data);

      REQUIRE(ret == 0);

      rstt::compressed_element& commitment_c =
          reinterpret_cast<rstt::compressed_element*>(commitments_data)[0];

      REQUIRE(commitment_c == expected_commitment_c);
    }

    sqccb::reset_backend_for_testing();
  }
}

TEST_CASE("run compute pedersen commitment tests") {
  test_pedersen_commitments_with_given_backend(SXT_NAIVE_BACKEND_CPU, 0, "naive cpu");
  test_pedersen_commitments_with_given_backend(SXT_NAIVE_BACKEND_GPU, 0, "naive gpu");
  test_pedersen_commitments_with_given_backend(SXT_PIPPENGER_BACKEND_CPU, 0, "pippenger cpu");

  test_pedersen_commitments_with_given_backend(SXT_NAIVE_BACKEND_CPU, 10,
                                               "naive cpu w/ precompute");
  test_pedersen_commitments_with_given_backend(SXT_NAIVE_BACKEND_GPU, 10,
                                               "naive gpu w/ precompute");
  test_pedersen_commitments_with_given_backend(SXT_PIPPENGER_BACKEND_CPU, 10,
                                               "pippenger cpu w/ precompute");
}

static void test_generators_with_given_backend(int backend, std::string backend_name) {
  SECTION(backend_name + " - zero num_generators will not error out") {
    const sxt_config config = {backend};
    REQUIRE(sxt_init(&config) == 0);

    uint64_t offset = 0;
    uint64_t num_generators = 0;

    int ret = sxt_get_generators(nullptr, num_generators, offset);

    REQUIRE(ret == 0);

    sqccb::reset_backend_for_testing();
  }

  SECTION(backend_name + " - non zero num_generators will not error out") {
    const sxt_config config = {backend};
    REQUIRE(sxt_init(&config) == 0);

    uint64_t num_generators = 10;
    sxt_ristretto generators[num_generators];

    int ret = sxt_get_generators(generators, num_generators, 0);

    REQUIRE(ret == 0);

    c21t::element_p3 expected_g[num_generators];
    for (size_t i = 0; i < num_generators; ++i) {
      sqcgn::compute_base_element(expected_g[i], i);

      REQUIRE(expected_g[i] == reinterpret_cast<c21t::element_p3*>(generators)[i]);
    }

    sqccb::reset_backend_for_testing();
  }

  SECTION(backend_name + " - nullptr generators pointer will error out") {
    const sxt_config config = {backend};
    REQUIRE(sxt_init(&config) == 0);

    uint64_t offset = 0;
    uint64_t num_generators = 3;

    int ret = sxt_get_generators(nullptr, num_generators, offset);

    REQUIRE(ret != 0);

    sqccb::reset_backend_for_testing();
  }

  SECTION(backend_name + " - computed generators are correct when offset is non zero") {
    const sxt_config config = {backend};
    REQUIRE(sxt_init(&config) == 0);

    uint64_t num_generators = 10;
    uint64_t offset_generators = 15;
    sxt_ristretto generators[num_generators];

    int ret = sxt_get_generators(generators, num_generators, offset_generators);

    REQUIRE(ret == 0);

    c21t::element_p3 expected_g[num_generators];
    for (size_t i = 0; i < num_generators; ++i) {
      sqcgn::compute_base_element(expected_g[i], i + offset_generators);

      REQUIRE(expected_g[i] == reinterpret_cast<c21t::element_p3*>(generators)[i]);
    }

    sqccb::reset_backend_for_testing();
  }
}

TEST_CASE("Fetching generators") {
  test_generators_with_given_backend(SXT_NAIVE_BACKEND_CPU, "naive cpu");
  test_generators_with_given_backend(SXT_NAIVE_BACKEND_GPU, "naive gpu");
  test_generators_with_given_backend(SXT_PIPPENGER_BACKEND_CPU, "pippenger cpu");
}

static void test_one_commitments_with_given_backend(int backend, std::string backend_name) {
  SECTION(backend_name + " - nullptr one_commitment pointer will error out") {
    const sxt_config config = {backend};
    REQUIRE(sxt_init(&config) == 0);
    int ret = sxt_get_one_commit(nullptr, 0);
    REQUIRE(ret != 0);
    sqccb::reset_backend_for_testing();
  }

  SECTION(backend_name + " - with valid results will not error out") {
    const sxt_config config = {backend};
    REQUIRE(sxt_init(&config) == 0);

    uint64_t num_generators = 10;
    sxt_ristretto generators[num_generators];
    int ret = sxt_get_generators(generators, num_generators, 0);
    REQUIRE(ret == 0);

    sxt_ristretto one_commitment;
    ret = sxt_get_one_commit(&one_commitment, 0);
    REQUIRE(ret == 0);
    REQUIRE(reinterpret_cast<c21t::element_p3*>(&one_commitment)[0] == c21cn::zero_p3_v);

    ret = sxt_get_one_commit(&one_commitment, 1);
    REQUIRE(ret == 0);
    REQUIRE(reinterpret_cast<c21t::element_p3*>(&one_commitment)[0] ==
            reinterpret_cast<c21t::element_p3*>(generators)[0]);

    c21t::element_p3 sum_gen_0_1;
    c21o::add(sum_gen_0_1, reinterpret_cast<c21t::element_p3*>(generators)[0],
              reinterpret_cast<c21t::element_p3*>(generators)[1]);

    ret = sxt_get_one_commit(&one_commitment, 2);
    REQUIRE(ret == 0);
    REQUIRE(reinterpret_cast<c21t::element_p3*>(&one_commitment)[0] ==
            reinterpret_cast<c21t::element_p3*>(&sum_gen_0_1)[0]);

    sqccb::reset_backend_for_testing();
  }
}

TEST_CASE("Fetching one commitments") {
  test_one_commitments_with_given_backend(SXT_NAIVE_BACKEND_CPU, "naive cpu");
  test_one_commitments_with_given_backend(SXT_NAIVE_BACKEND_GPU, "naive gpu");
  test_one_commitments_with_given_backend(SXT_PIPPENGER_BACKEND_CPU, "pippenger cpu");
}
