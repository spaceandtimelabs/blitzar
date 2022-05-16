#include "sxt/seqcommit/cbindings/pedersen.h"

#include <array>
#include <algorithm>

#include "sxt/base/test/unit_test.h"

TEST_CASE("run pedersen tests") {
    const uint32_t num_sequences = 3;
    sxt_commitment commitments[num_sequences];

    const uint64_t n1 = 3;
    const uint8_t n1_num_bytes = 1;
    uint8_t data_bytes_1[n1_num_bytes * n1];
    sxt_dense_sequence_descriptor valid_seq_descriptor1 = {
        n1_num_bytes, // number bytes
        n1, // number rows
        data_bytes_1 // data pointer
    };

    const uint64_t n2 = 2;
    const uint8_t n2_num_bytes = 4;
    uint8_t data_bytes_2[n2_num_bytes * n1];
    sxt_dense_sequence_descriptor valid_seq_descriptor2 = {
        n2_num_bytes,
        n2, // number rows
        data_bytes_2 // data pointer
    };

    const uint64_t n3 = 1;
    const uint8_t n3_num_bytes = 32;
    uint8_t data_bytes_3[n3_num_bytes * n1];
    sxt_dense_sequence_descriptor valid_seq_descriptor3 = {
        n3_num_bytes, // number bytes
        n3, // number rows
        data_bytes_3 // data pointer
    };

    const sxt_sequence_descriptor valid_descriptors[num_sequences] = {
        {SXT_DENSE_SEQUENCE_TYPE, valid_seq_descriptor1},
        {SXT_DENSE_SEQUENCE_TYPE, valid_seq_descriptor2},
        {SXT_DENSE_SEQUENCE_TYPE, valid_seq_descriptor3},
    };

    SECTION("non initialized library will error out") {
        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            valid_descriptors
        );

        REQUIRE(ret != 0);
    }

    SECTION("multiple calls to the library initialization will not error out") {
        const sxt_config config = {SXT_BACKEND_CPU};

        REQUIRE(sxt_init(&config) == 0);

        REQUIRE(sxt_init(&config) == 0);
    }

    SECTION("initialize library with GPU backend will not error out") {
        const sxt_config config = {SXT_BACKEND_GPU};

        REQUIRE(sxt_init(&config) == 0);
    }

    SECTION("incorrect input to the library initialization will error out") {
        REQUIRE(sxt_init(NULL) != 0);

        const sxt_config config = {SXT_BACKEND_CPU + SXT_BACKEND_GPU};

        REQUIRE(sxt_init(&config) != 0);
    }

    SECTION("null commitment pointers will error out") {
        const sxt_config config = {SXT_BACKEND_CPU};

        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            nullptr,
            num_sequences,
            valid_descriptors
        );

        REQUIRE(ret != 0);
    }

    SECTION("null value_sequences will error out") {
        const sxt_config config = {SXT_BACKEND_CPU};

        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            nullptr
        );

        REQUIRE(ret != 0);
    }

    SECTION("zero sequences will not error out") {
        const sxt_config config = {SXT_BACKEND_CPU};

        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            0,
            valid_descriptors
        );

        REQUIRE(ret == 0);
    }

    const uint8_t zero_length_num_bytes = 4;
    uint8_t zero_length_data_bytes[zero_length_num_bytes * n1];
    sxt_dense_sequence_descriptor zero_length_seq_descriptor = {
        zero_length_num_bytes, // number bytes
        0, // number rows
        zero_length_data_bytes // data pointer
    };
    const sxt_sequence_descriptor invalid_descriptors[num_sequences] = {
        {SXT_DENSE_SEQUENCE_TYPE, valid_seq_descriptor1},
        {SXT_DENSE_SEQUENCE_TYPE, zero_length_seq_descriptor},
        {SXT_DENSE_SEQUENCE_TYPE, valid_seq_descriptor3},
    };
    
    SECTION("zero length commitments will not error out on cpu") {
        const sxt_config config = {SXT_BACKEND_CPU};

        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            invalid_descriptors
        );

        REQUIRE(ret == 0);

        std::array<uint8_t, 32> res_array;
        std::array<uint8_t, 32> req_array = {0};

        std::copy(commitments[1].ristretto_bytes,
                commitments[1].ristretto_bytes + 32, res_array.data());

        REQUIRE(res_array == req_array);
    }

    SECTION("zero length commitments will not error out on gpu") {
        const sxt_config config = {SXT_BACKEND_GPU};

        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            invalid_descriptors
        );

        REQUIRE(ret == 0);

        std::array<uint8_t, 32> res_array;
        std::array<uint8_t, 32> req_array = {0};

        std::copy(commitments[1].ristretto_bytes,
            commitments[1].ristretto_bytes + 32, res_array.data());

        REQUIRE(res_array == req_array);
    }

    SECTION("descriptor with invalid type will error out") {
        const sxt_config config = {SXT_BACKEND_CPU};

        const sxt_sequence_descriptor invalid_descriptors[num_sequences] = {
            {SXT_DENSE_SEQUENCE_TYPE, valid_seq_descriptor1},
            {SXT_DENSE_SEQUENCE_TYPE + SXT_DENSE_SEQUENCE_TYPE, valid_seq_descriptor2},
            {SXT_DENSE_SEQUENCE_TYPE, valid_seq_descriptor3},
        };

        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            invalid_descriptors
        );

        REQUIRE(ret == 0);
    }

    SECTION("out of range (< 1 or > 32) element_nbytes will error out") {
        const sxt_config config = {SXT_BACKEND_CPU};
        
        const uint8_t invalid_num_bytes1 = 0;
        uint8_t invalid_data_bytes1[invalid_num_bytes1 * n1];
        sxt_dense_sequence_descriptor invalid_seq_descriptor1 = {
            invalid_num_bytes1, // number bytes
            n1, // number rows
            invalid_data_bytes1 // data pointer
        };

        const uint8_t invalid_num_bytes12 = 33;
        uint8_t data_bytes_2[invalid_num_bytes12 * n1];
        sxt_dense_sequence_descriptor invalid_seq_descriptor2 = {
            invalid_num_bytes12, // number bytes
            n1, // number rows
            data_bytes_2 // data pointer
        };
        
        const sxt_sequence_descriptor invalid_descriptors[num_sequences] = {
            {SXT_DENSE_SEQUENCE_TYPE, invalid_seq_descriptor1},
            {SXT_DENSE_SEQUENCE_TYPE, invalid_seq_descriptor2},
            {SXT_DENSE_SEQUENCE_TYPE, valid_seq_descriptor3},
        };

        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            invalid_descriptors
        );

        REQUIRE(ret != 0);
    }

    SECTION("null element data pointer will error out") {
        const sxt_config config = {SXT_BACKEND_CPU};
        
        sxt_dense_sequence_descriptor invalid_seq_descriptor1 = {
            n1_num_bytes, // number bytes
            n1, // number rows
            nullptr // null data pointer
        };

        const sxt_sequence_descriptor invalid_descriptors[num_sequences] = {
            {SXT_DENSE_SEQUENCE_TYPE, invalid_seq_descriptor1},
            {SXT_DENSE_SEQUENCE_TYPE, valid_seq_descriptor2},
            {SXT_DENSE_SEQUENCE_TYPE, valid_seq_descriptor3},
        };

        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            invalid_descriptors
        );

        REQUIRE(ret != 0);
    }

    SECTION("correct library initialization and input will not error out") {
        const sxt_config config = {SXT_BACKEND_CPU};

        REQUIRE(sxt_init(&config) == 0);

        int ret = sxt_compute_pedersen_commitments(
            commitments,
            num_sequences,
            valid_descriptors
        );

        REQUIRE(ret == 0);
    }
}
