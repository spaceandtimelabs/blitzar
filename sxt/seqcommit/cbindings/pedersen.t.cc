#include "sxt/seqcommit/cbindings/pedersen.h"

#include <array>
#include <string>
#include <algorithm>

#include "sxt/base/test/unit_test.h"
#include "sxt/ristretto/operation/add.h"
#include "sxt/ristretto/random/element.h"
#include "sxt/seqcommit/generator/base_element.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/ristretto/operation/scalar_multiply.h"

using namespace sxt;
using namespace sxt::sqccb;

TEST_CASE("run pedersen initialization and finalize tests") {
    sxt_ristretto_element commitments[1];
    sxt_sequence_descriptor valid_descriptors[1];
    
    SECTION("incorrect input to the initialization will error out") {
        const sxt_config config = {100000};

        REQUIRE(sxt_init(&config) != 0);
    }

    SECTION("not initialized library will error out") {
        int ret = sxt_compute_pedersen_commitments(
            commitments,
            0,
            valid_descriptors
        );

        REQUIRE(ret != 0);
    }

    SECTION("correct naive cpu backend will not error out") {
        const sxt_config config = {SXT_NAIVE_BACKEND_CPU};
        REQUIRE(sxt_init(&config) == 0);
        
        int ret = sxt_compute_pedersen_commitments(
            commitments,
            0,
            valid_descriptors
        );

        REQUIRE(ret == 0);

        REQUIRE(reset_backend_for_testing() == 0);
    }

    SECTION("correct naive gpu backend will not error out") {
        const sxt_config config = {SXT_NAIVE_BACKEND_GPU};
        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            0,
            valid_descriptors
        );

        REQUIRE(ret == 0);

        REQUIRE(reset_backend_for_testing() == 0);
    }

    SECTION("correct pippenger cpu backend will not error out") {
        const sxt_config config = {SXT_PIPPENGER_BACKEND_CPU};
        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            0,
            valid_descriptors
        );

        REQUIRE(ret == 0);

        REQUIRE(reset_backend_for_testing() == 0);
    }
}

static void test_pedersen_commitments_with_given_backend(
        int backend, std::string backend_name) {

    ////////////////////////////////////////////////////////////////
    // sxt_compute_pedersen_commitments
    ////////////////////////////////////////////////////////////////
    SECTION(backend_name +
        " - We can compute commitments without providing any generator") {

        SECTION("correct initialization and input will not error out") {
            const sxt_config config = {backend};
            REQUIRE(sxt_init(&config) == 0);
            
            uint8_t data[33];
            sxt_ristretto_element commitment;
            sxt_sequence_descriptor valid_seq_descriptor = {
                .element_nbytes = 1, // number bytes
                .n = 1, // number rows
                .data = data, // data pointer
                .indices = nullptr
            };

            int ret = sxt_compute_pedersen_commitments(
                &commitment,
                1,
                &valid_seq_descriptor
            );

            REQUIRE(ret == 0);

            REQUIRE(reset_backend_for_testing() == 0);
        }

        SECTION("null commitment pointers will error out") {
            const sxt_config config = {backend};
            REQUIRE(sxt_init(&config) == 0);

            uint8_t data[33];
            sxt_sequence_descriptor valid_descriptor = {
                .element_nbytes = 1, // number bytes
                .n = 1, // number rows
                .data = data, // data pointer
                nullptr
            };

            int ret = sxt_compute_pedersen_commitments(
                nullptr,
                1,
                &valid_descriptor
            );

            REQUIRE(ret != 0);

            REQUIRE(reset_backend_for_testing() == 0);
        }

        SECTION("null value_sequences will error out") {
            const sxt_config config = {backend};
            REQUIRE(sxt_init(&config) == 0);
            
            sxt_ristretto_element commitment;
            int ret = sxt_compute_pedersen_commitments(
                &commitment,
                1,
                nullptr
            );

            REQUIRE(ret != 0);

            REQUIRE(reset_backend_for_testing() == 0);
        }

        SECTION("zero sequences will not error out") {
            const sxt_config config = {backend};
            REQUIRE(sxt_init(&config) == 0);

            sxt_ristretto_element commitment;
            int ret = sxt_compute_pedersen_commitments(
                &commitment,
                0,
                nullptr
            );

            REQUIRE(ret == 0);

            REQUIRE(reset_backend_for_testing() == 0);
        }
        
        SECTION("zero length commitments will not error out") {
            const sxt_config config = {backend};
            REQUIRE(sxt_init(&config) == 0);

            uint8_t data[33];
            sxt_ristretto_element commitment;
            sxt_sequence_descriptor zero_length_seq_descriptor = {
                .element_nbytes = 1, // number bytes
                .n = 0, // number rows
                .data = data, // data pointer
                nullptr
            };

            int ret = sxt_compute_pedersen_commitments(
                &commitment,
                1,
                &zero_length_seq_descriptor
            );

            REQUIRE(ret == 0);

            REQUIRE(rstt::compressed_element() == 
                    reinterpret_cast<rstt::compressed_element *>(&commitment)[0]);

            REQUIRE(reset_backend_for_testing() == 0);
        }

        SECTION("out of range (< 1 or > 32) element_nbytes will error out") {
            const sxt_config config = {backend};
            REQUIRE(sxt_init(&config) == 0);
            
            SECTION("element_nbytes == 0 (< 1) error out") {
                uint8_t data[33];
                sxt_ristretto_element commitment;
                sxt_sequence_descriptor invalid_descriptors = {
                    .element_nbytes = 0, // number bytes
                    .n = 1, // number rows
                    .data = data, // data pointer
                    .indices = nullptr
                };
                
                int ret = sxt_compute_pedersen_commitments(
                    &commitment,
                    1,
                    &invalid_descriptors
                );

                REQUIRE(ret != 0);
            }

            SECTION("element_nbytes == 33 (> 32) error out") {
                uint8_t data[33];
                sxt_ristretto_element commitment;
                sxt_sequence_descriptor invalid_descriptors = {
                    .element_nbytes = 33, // number bytes
                    .n = 1, // number rows
                    .data = data, // data pointer
                    .indices = nullptr
                };
                
                int ret = sxt_compute_pedersen_commitments(
                    &commitment,
                    1,
                    &invalid_descriptors
                );

                REQUIRE(ret != 0);
            }

            REQUIRE(reset_backend_for_testing() == 0);
        }

        SECTION("null element data pointer will error out") {
            const sxt_config config = {backend};
            REQUIRE(sxt_init(&config) == 0);
            
            sxt_ristretto_element commitment;
            sxt_sequence_descriptor invalid_descriptors = {
                .element_nbytes = 1, // number bytes
                .n = 1, // number rows
                .data = nullptr, // null data pointer
                .indices = nullptr
            };

            int ret = sxt_compute_pedersen_commitments(
                &commitment,
                1,
                &invalid_descriptors
            );

            REQUIRE(ret != 0);

            REQUIRE(reset_backend_for_testing() == 0);
        }

        SECTION("We can multiply and add two commitments together") {
            const sxt_config config = {backend};
            REQUIRE(sxt_init(&config) == 0);

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
                    element_nbytes, num_rows,
                    reinterpret_cast<const uint8_t *>(query[i]), nullptr
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

                rstt::compressed_element p =
                    reinterpret_cast<rstt::compressed_element *>(commitments_data)[0];

                rstt::compressed_element q =
                    reinterpret_cast<rstt::compressed_element *>(commitments_data)[1];;

                rsto::scalar_multiply(
                    p, multiplicative_constant, p);  // h_i = a_i * g_i

                rsto::add(p, p, q);

                rstt::compressed_element expected_commitment_c =
                    reinterpret_cast<rstt::compressed_element *>(commitments_data)[2];

                // verify that result is not null
                REQUIRE(rstt::compressed_element() != p);

                REQUIRE(p == expected_commitment_c);
            }

            REQUIRE(reset_backend_for_testing() == 0);
        }

        SECTION("We can compute sparse commitments") {
            const sxt_config config = {backend};
            REQUIRE(sxt_init(&config) == 0);

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
                element_nbytes, 11, 
                reinterpret_cast<const uint8_t *>(dense_data),
                nullptr
            };

            sxt_sequence_descriptor sparse_descriptor = {
                element_nbytes, 5, 
                reinterpret_cast<const uint8_t *>(sparse_data),
                sparse_indices
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

                rstt::compressed_element &dense_commitment =
                    reinterpret_cast<rstt::compressed_element *>(commitments_data)[0];

                rstt::compressed_element &sparse_commitment =
                    reinterpret_cast<rstt::compressed_element *>(commitments_data)[1];

                // verify that result is not null
                REQUIRE(rstt::compressed_element() != dense_commitment);

                REQUIRE(dense_commitment == sparse_commitment);
            }

            REQUIRE(reset_backend_for_testing() == 0);
        }

        SECTION("we can compute dense commitments as sparse commitments") {
            const sxt_config config = {backend};
            REQUIRE(sxt_init(&config) == 0);

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
                element_nbytes, 11,
                reinterpret_cast<const uint8_t *>(dense_data), nullptr
            };

            sxt_sequence_descriptor sparse_descriptor = {
                element_nbytes, 11,
                reinterpret_cast<const uint8_t *>(sparse_data), sparse_indices
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

                rstt::compressed_element &dense_commitment =
                    reinterpret_cast<rstt::compressed_element *>(commitments_data)[0];

                rstt::compressed_element &sparse_commitment =
                    reinterpret_cast<rstt::compressed_element *>(commitments_data)[1];

                // verify that result is not null
                REQUIRE(rstt::compressed_element() != dense_commitment);
                
                REQUIRE(dense_commitment == sparse_commitment);
            }

            REQUIRE(reset_backend_for_testing() == 0);
        }
    }

    ////////////////////////////////////////////////////////////////
    // sxt_compute_pedersen_commitments_with_generators
    ////////////////////////////////////////////////////////////////

    SECTION(backend_name +
        " - We can compute commitments with provided generators") {
        const sxt_config config = {backend};
        REQUIRE(sxt_init(&config) == 0);

        const uint64_t num_rows = 4;
        const uint64_t num_sequences = 1;
        const uint8_t element_nbytes = sizeof(int);

        sxt_ristretto_element generators_data[num_rows];
        sxt_ristretto_element commitments_data[num_sequences];

        const uint32_t query[num_rows] = {2000, 7500, 5000, 1500};
        rstt::compressed_element expected_g;

        for (uint64_t i = 0; i < num_rows; ++i) {
            rstt::compressed_element &g_i =
                reinterpret_cast<rstt::compressed_element *>(generators_data)[i];

            sqcgn::compute_compressed_base_element(g_i, query[i]);

            rstt::compressed_element h;

            rsto::scalar_multiply(h, query[i], g_i);

            rsto::add(expected_g, expected_g, h);
        }
        
        rstt::compressed_element expected_commitment_c = expected_g;

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

        SECTION("passing valid generators will not error out") {
            int ret = sxt_compute_pedersen_commitments_with_generators(
                commitments_data,
                num_sequences,
                valid_descriptors,
                generators_data
            );

            REQUIRE(ret == 0);
            
            rstt::compressed_element &commitment_c =
                reinterpret_cast<rstt::compressed_element *>(commitments_data)[0];

            REQUIRE(commitment_c == expected_commitment_c);
        }

        REQUIRE(reset_backend_for_testing() == 0);
    }
}

TEST_CASE("run compute pedersen commitment tests") {
    test_pedersen_commitments_with_given_backend(
                SXT_NAIVE_BACKEND_CPU, "naive cpu");
    test_pedersen_commitments_with_given_backend(
                SXT_NAIVE_BACKEND_GPU, "naive gpu");
    test_pedersen_commitments_with_given_backend(
                SXT_PIPPENGER_BACKEND_CPU, "pippenger cpu");
}

static void test_generators_with_given_backend(int backend, std::string backend_name) {
    SECTION(backend_name +
        " - zero num_generators will not error out") {
        const sxt_config config = {backend};
        REQUIRE(sxt_init(&config) == 0);

        uint64_t offset = 0;
        uint64_t num_generators = 0;

        int ret = sxt_get_generators(
            nullptr,
            num_generators,
            offset
        );

        REQUIRE(ret == 0);

        REQUIRE(reset_backend_for_testing() == 0);
    }

    SECTION(backend_name +
        " - non zero num_generators will not error out") {
        const sxt_config config = {backend};
        REQUIRE(sxt_init(&config) == 0);

        uint64_t num_generators = 10;
        sxt_ristretto_element generators[num_generators];

        int ret = sxt_get_generators(
            generators,
            num_generators,
            0
        );

        REQUIRE(ret == 0);

        rstt::compressed_element expected_g[num_generators];
        for (size_t i = 0; i < num_generators; ++i) {
            sqcgn::compute_compressed_base_element(expected_g[i], i);

            REQUIRE(expected_g[i] == 
                reinterpret_cast<rstt::compressed_element *>(generators)[i]);
        }

        REQUIRE(reset_backend_for_testing() == 0);
    }

    SECTION(backend_name +
        " - nullptr generators pointer will error out") {
        const sxt_config config = {backend};
        REQUIRE(sxt_init(&config) == 0);

        uint64_t offset = 0;
        uint64_t num_generators = 3;

        int ret = sxt_get_generators(
            nullptr,
            num_generators,
            offset
        );

        REQUIRE(ret != 0);

        REQUIRE(reset_backend_for_testing() == 0);
    }

    SECTION(backend_name +
        " - computed generators are correct when offset is non zero") {
        const sxt_config config = {backend};
        REQUIRE(sxt_init(&config) == 0);

        uint64_t num_generators = 10;
        uint64_t offset_generators = 15;
        sxt_ristretto_element generators[num_generators];

        int ret = sxt_get_generators(
            generators,
            num_generators,
            offset_generators
        );

        REQUIRE(ret == 0);

        rstt::compressed_element expected_g[num_generators];
        for (size_t i = 0; i < num_generators; ++i) {
            sqcgn::compute_compressed_base_element(
                        expected_g[i], i + offset_generators);

            REQUIRE(expected_g[i] == 
                reinterpret_cast<rstt::compressed_element *>(generators)[i]);
        }

        REQUIRE(reset_backend_for_testing() == 0);
    }
}

TEST_CASE("Fetching generators") {
    test_generators_with_given_backend(SXT_NAIVE_BACKEND_CPU, "naive cpu");
    test_generators_with_given_backend(SXT_NAIVE_BACKEND_GPU, "naive gpu");
    test_generators_with_given_backend(SXT_PIPPENGER_BACKEND_CPU, "pippenger cpu");
}
