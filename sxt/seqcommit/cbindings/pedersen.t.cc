#include "sxt/seqcommit/cbindings/pedersen.h"

#include <array>
#include <algorithm>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/seqcommit/generator/base_element.h"
#include "sxt/curve21/ristretto/byte_conversion.h"
#include "sxt/curve21/operation/scalar_multiply.h"

TEST_CASE("run compute pedersen commitment tests") {
    const uint32_t num_sequences = 3;
    sxt_ristretto_element commitments[num_sequences];

    const uint64_t n1 = 3;
    const uint8_t n1_num_bytes = 1;
    uint8_t data_bytes_1[n1_num_bytes * n1];
    sxt_sequence_descriptor valid_seq_descriptor1 = {
        n1_num_bytes, // number bytes
        n1, // number rows
        data_bytes_1, // data pointer
        nullptr
    };

    const uint64_t n2 = 2;
    const uint8_t n2_num_bytes = 4;
    uint8_t data_bytes_2[n2_num_bytes * n1];
    sxt_sequence_descriptor valid_seq_descriptor2 = {
        n2_num_bytes,
        n2, // number rows
        data_bytes_2, // data pointer
        nullptr
    };

    const uint64_t n3 = 1;
    const uint8_t n3_num_bytes = 32;
    uint8_t data_bytes_3[n3_num_bytes * n1];
    sxt_sequence_descriptor valid_seq_descriptor3 = {
        n3_num_bytes, // number bytes
        n3, // number rows
        data_bytes_3, // data pointer
        nullptr
    };

    const sxt_sequence_descriptor valid_descriptors[num_sequences] = {
        valid_seq_descriptor1,
        valid_seq_descriptor2,
        valid_seq_descriptor3,
    };

    SECTION("non initialized library will error out") {
        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            valid_descriptors
        );

        REQUIRE(ret != 0);
    }

    ////////////////////////////////////////////////////////////////
    // sxt_compute_pedersen_commitments
    ////////////////////////////////////////////////////////////////

    SECTION("incorrect input to the library initialization will error out") {
        const sxt_config config = {SXT_BACKEND_CPU + SXT_BACKEND_GPU};

        REQUIRE(sxt_init(&config) != 0);
    }

    SECTION("correct library initialization and input will not error out") {
        const sxt_config config = {SXT_BACKEND_GPU};

        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            valid_descriptors
        );

        REQUIRE(ret == 0);
    }

    SECTION("null commitment pointers will error out") {
        int ret = sxt_compute_pedersen_commitments(
            nullptr,
            num_sequences,
            valid_descriptors
        );

        REQUIRE(ret != 0);
    }

    SECTION("null value_sequences will error out") {
        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            nullptr
        );

        REQUIRE(ret != 0);
    }

    SECTION("zero sequences will not error out") {
        int ret = sxt_compute_pedersen_commitments(
            commitments,
            0,
            valid_descriptors
        );

        REQUIRE(ret == 0);
    }

    const uint8_t zero_length_num_bytes = 4;
    uint8_t zero_length_data_bytes[zero_length_num_bytes * n1];
    sxt_sequence_descriptor zero_length_seq_descriptor = {
        zero_length_num_bytes, // number bytes
        0, // number rows
        zero_length_data_bytes, // data pointer
        nullptr
    };
    const sxt_sequence_descriptor invalid_descriptors[num_sequences] = {
        valid_seq_descriptor1,
        zero_length_seq_descriptor,
        valid_seq_descriptor3,
    };
    
    std::array<uint8_t, 32> res_array;
    std::array<uint8_t, 32> req_array = {};
    
    SECTION("zero length commitments will not error out on cpu") {
        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            invalid_descriptors
        );

        REQUIRE(ret == 0);

        std::copy(commitments[1].ristretto_bytes,
                commitments[1].ristretto_bytes + 32, res_array.data());

        REQUIRE(res_array == req_array);
    }

    SECTION("zero length commitments will not error out on gpu") {
        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            invalid_descriptors
        );

        REQUIRE(ret == 0);

        std::copy(commitments[1].ristretto_bytes,
            commitments[1].ristretto_bytes + 32, res_array.data());

        REQUIRE(res_array == req_array);
    }

    SECTION("out of range (< 1 or > 32) element_nbytes will error out") {
        const uint8_t invalid_num_bytes1 = 0;
        uint8_t invalid_data_bytes1[invalid_num_bytes1 * n1];
        sxt_sequence_descriptor invalid_seq_descriptor1 = {
            invalid_num_bytes1, // number bytes
            n1, // number rows
            invalid_data_bytes1, // data pointer
            nullptr
        };

        const uint8_t invalid_num_bytes12 = 33;
        uint8_t data_bytes_2[invalid_num_bytes12 * n1];
        sxt_sequence_descriptor invalid_seq_descriptor2 = {
            invalid_num_bytes12, // number bytes
            n1, // number rows
            data_bytes_2, // data pointer
            nullptr
        };
        
        const sxt_sequence_descriptor invalid_descriptors[num_sequences] = {
            invalid_seq_descriptor1,
            invalid_seq_descriptor2,
            valid_seq_descriptor3,
        };

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            invalid_descriptors
        );

        REQUIRE(ret != 0);
    }

    SECTION("null element data pointer will error out") {
        sxt_sequence_descriptor invalid_seq_descriptor1 = {
            n1_num_bytes, // number bytes
            n1, // number rows
            nullptr // null data pointer
        };

        const sxt_sequence_descriptor invalid_descriptors[num_sequences] = {
            invalid_seq_descriptor1,
            valid_seq_descriptor2,
            valid_seq_descriptor3,
        };

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            invalid_descriptors
        );

        REQUIRE(ret != 0);
    }

    SECTION("We can multiply and add two commitments together") {
        const uint64_t num_rows = 4;
        const uint64_t num_sequences = 3;
        const uint8_t element_nbytes = sizeof(int);
        const unsigned int multiplicative_constant = 52;

        sxt_ristretto_element commitments_data[num_sequences];

        const int query[num_sequences][num_rows] = {
            {2000, 7500, 5000, 1500},
            {5000, 0, 400000, 10},
            {
                multiplicative_constant * 2000 + 5000,
                multiplicative_constant * 7500 + 0,
                multiplicative_constant * 5000 + 400000,
                multiplicative_constant * 1500 + 10
            }
        };

        sxt_sequence_descriptor valid_descriptors[num_sequences];

        // populating sequence object
        for (uint64_t i = 0; i < num_sequences; ++i) {
            sxt_sequence_descriptor descriptor = {
                element_nbytes, num_rows, reinterpret_cast<const uint8_t *>(query[i]), nullptr
            };

            valid_descriptors[i] = descriptor;
        }

        SECTION("c = 52 * a + b ==> commit_c = 52 * commit_a + commit_b") {
            int ret = sxt_compute_pedersen_commitments(
                commitments_data,
                num_sequences,
                valid_descriptors
            );

            REQUIRE(ret == 0);

            sxt::sqcb::commitment commitment_c;

            sxt::c21t::element_p3 p, q;

            sxt::c21rs::from_bytes(p, commitments_data[0].ristretto_bytes);

            sxt::c21rs::from_bytes(q, commitments_data[1].ristretto_bytes);

            sxt::c21o::scalar_multiply(p, multiplicative_constant,
                                    p);  // h_i = a_i * g_i

            sxt::c21o::add(p, p, q);

            sxt::c21rs::to_bytes(commitment_c.data(), p);

            sxt::sqcb::commitment &expected_commitment_c =
                reinterpret_cast<sxt::sqcb::commitment *>(commitments_data)[2];

            // verify that result is not null
            REQUIRE(sxt::sqcb::commitment() != commitment_c);

            REQUIRE(commitment_c == expected_commitment_c);
        }
    }

    SECTION("We can compute sparse commitments") {
        const uint64_t num_sequences = 2;
        const uint8_t element_nbytes = sizeof(int);

        sxt_ristretto_element commitments_data[num_sequences];

        const int dense_data[11] = {
            1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0
        };

        const int sparse_data[5] = {
            1, 2, 3, 4, 9
        };

        const uint64_t sparse_indices[5] = {
            0, 2, 4, 5, 9
        };

        sxt_sequence_descriptor dense_descriptor = {
            element_nbytes, 11, reinterpret_cast<const uint8_t *>(dense_data), nullptr
        };

        sxt_sequence_descriptor sparse_descriptor = {
            element_nbytes, 5, reinterpret_cast<const uint8_t *>(sparse_data), sparse_indices
        };

        sxt_sequence_descriptor descriptors[num_sequences] = {
            dense_descriptor, sparse_descriptor
        };

        SECTION("sparse_commitments == dense_commitment") {
            int ret = sxt_compute_pedersen_commitments(
                commitments_data,
                num_sequences,
                descriptors
            );

            REQUIRE(ret == 0);

            sxt::sqcb::commitment &dense_commitment =
                reinterpret_cast<sxt::sqcb::commitment *>(commitments_data)[0];

            sxt::sqcb::commitment &sparse_commitment =
                reinterpret_cast<sxt::sqcb::commitment *>(commitments_data)[1];

            // verify that result is not null
            REQUIRE(sxt::sqcb::commitment() != dense_commitment);

            REQUIRE(dense_commitment == sparse_commitment);
        }
    }

    SECTION("We can compute dense commitments as sparse commitments") {
        const uint64_t num_sequences = 2;
        const uint8_t element_nbytes = sizeof(int);

        sxt_ristretto_element commitments_data[num_sequences];

        const int dense_data[11] = {
            1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0
        };

        const int sparse_data[11] = {
            1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0
        };

        const uint64_t sparse_indices[11] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        };

        sxt_sequence_descriptor dense_descriptor = {
            element_nbytes, 11, reinterpret_cast<const uint8_t *>(dense_data), nullptr
        };

        sxt_sequence_descriptor sparse_descriptor = {
            element_nbytes, 11, reinterpret_cast<const uint8_t *>(sparse_data), sparse_indices
        };

        sxt_sequence_descriptor descriptors[num_sequences] = {
            dense_descriptor, sparse_descriptor
        };

        SECTION("dense_commitment == dense_commitment as sparse") {
            int ret = sxt_compute_pedersen_commitments(
                commitments_data,
                num_sequences,
                descriptors
            );

            REQUIRE(ret == 0);

            sxt::sqcb::commitment &dense_commitment =
                reinterpret_cast<sxt::sqcb::commitment *>(commitments_data)[0];

            sxt::sqcb::commitment &sparse_commitment =
                reinterpret_cast<sxt::sqcb::commitment *>(commitments_data)[1];

            // verify that result is not null
            REQUIRE(sxt::sqcb::commitment() != dense_commitment);
            
            REQUIRE(dense_commitment == sparse_commitment);
        }
    }

    ////////////////////////////////////////////////////////////////
    // sxt_compute_pedersen_commitments_with_generators
    ////////////////////////////////////////////////////////////////

    SECTION("We can multiply and add two commitments together using"
         "the sxt_compute_pedersen_commitments_with_generators function") {
        const uint64_t num_rows = 4;
        const uint64_t num_sequences = 1;
        const uint8_t element_nbytes = sizeof(int);

        sxt_ristretto_element generators_data[num_rows];
        sxt_ristretto_element commitments_data[num_sequences];

        const int query[num_rows] = {2000, 7500, 5000, 1500};
        sxt::c21t::element_p3 expected_g = sxt::c21cn::zero_p3_v;

        for (uint64_t i = 0; i < num_rows; ++i) {
            sxt::c21t::element_p3 g_i;
            sxt::sqcgn::compute_base_element(g_i, query[i]);

            sxt::c21rs::to_bytes(generators_data[i].ristretto_bytes, g_i);
            sxt::c21rs::from_bytes(g_i, generators_data[i].ristretto_bytes);

            sxt::c21t::element_p3 h;
            sxt::c21o::scalar_multiply(
                h,
                sxt::basct::cspan<uint8_t>{
                    reinterpret_cast<const uint8_t*>(query + i), sizeof(int)
                },
                g_i
            );

            sxt::c21o::add(expected_g, expected_g, h);
        }
        
        sxt::sqcb::commitment expected_commitment_c;
        sxt::c21rs::to_bytes(expected_commitment_c.data(), expected_g);

        sxt_sequence_descriptor valid_descriptors[num_sequences];

        // populating sequence object
        for (uint64_t i = 0; i < num_sequences; ++i) {
            sxt_sequence_descriptor descriptor = {
                element_nbytes, num_rows, reinterpret_cast<const uint8_t *>(query), nullptr
            };

            valid_descriptors[i] = descriptor;
        }

        SECTION("passing null generators will error out") {
            int ret = sxt_compute_pedersen_commitments_with_generators(
                commitments_data,
                num_sequences,
                valid_descriptors,
                nullptr
            );

            REQUIRE(ret != 0);
        }

        SECTION("passing valid generators will be correct") {
            int ret = sxt_compute_pedersen_commitments_with_generators(
                commitments_data,
                num_sequences,
                valid_descriptors,
                generators_data
            );

            REQUIRE(ret == 0);
            
            sxt::sqcb::commitment &commitment_c =
                reinterpret_cast<sxt::sqcb::commitment *>(commitments_data)[0];

            REQUIRE(commitment_c == expected_commitment_c);
        }
    }
}

TEST_CASE("run get pedersen generator tests") {
    SECTION("zero generators will not error out") {
        uint64_t offset = 0;
        uint64_t num_generators = 0;

        int ret = sxt_get_generators(
            nullptr,
            num_generators,
            offset
        );

        REQUIRE(ret == 0);
    }

    SECTION("non zero generators will not error out") {
        uint64_t offset = 0;
        uint64_t num_generators = 3;
        sxt_ristretto_element generators[num_generators];

        int ret = sxt_get_generators(
            generators,
            num_generators,
            offset
        );

        REQUIRE(ret == 0);
    }

    SECTION("non zero generators and nullptr will error out") {
        uint64_t offset = 0;
        uint64_t num_generators = 3;

        int ret = sxt_get_generators(
            nullptr,
            num_generators,
            offset
        );

        REQUIRE(ret != 0);
    }

    SECTION("we can verify that computed generators are correct when offset is non zero") {
      sxt::c21t::element_p3 expected_g_0, expected_g_1;
      uint64_t num_generators = 2;
      uint64_t offset_generators = 15;
      sxt::sqcgn::compute_base_element(expected_g_0, 0 + offset_generators);
      sxt::sqcgn::compute_base_element(expected_g_1, 1 + offset_generators);

      sxt_ristretto_element generators[num_generators];

      int ret = sxt_get_generators(
        generators,
        num_generators,
        offset_generators
      );

      REQUIRE(ret == 0);

      sxt::sqcb::commitment expected_commit_0, expected_commit_1;
      sxt::c21rs::to_bytes(expected_commit_0.data(), expected_g_0);
      sxt::c21rs::to_bytes(expected_commit_1.data(), expected_g_1);

      REQUIRE(reinterpret_cast<sxt::sqcb::commitment *>(generators)[0]
                 == expected_commit_0);
      REQUIRE(reinterpret_cast<sxt::sqcb::commitment *>(generators)[1]
                 == expected_commit_1);
    }
}
