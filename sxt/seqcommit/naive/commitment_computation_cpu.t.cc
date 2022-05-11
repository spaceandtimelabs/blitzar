#include "sxt/seqcommit/naive/commitment_computation_cpu.h"

#include "sxt/seqcommit/naive/fill_data.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/curve21/ristretto/byte_conversion.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/seqcommit/base/base_element.h"
#include "sxt/curve21/operation/scalar_multiply.h"

using namespace sxt;
using namespace sxt::sqcnv;

TEST_CASE("Test 1 - We can add two commitments together") {
    const uint64_t num_rows = 4;
    const uint64_t num_columns = 3;
    const uint8_t element_nbytes = sizeof(int);

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_columns);

    const int query[num_columns][num_rows] = {
        {       2000,     7500,          5000,      1500},
        {       5000,        0,        400000,        10},
        {2000 + 5000, 7500 + 0, 5000 + 400000, 1500 + 10}
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_columns; ++i) {
        sequences[i].n = num_rows;
        sequences[i].data = (const uint8_t *) query[i];
        sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("Verifying if addion property holds (c = a + b ==> commit_c = commit_a + commit_b)") {
        sqcnv::compute_commitments_cpu(commitments, value_sequences);
        
        sqcb::commitment commitment_c;

        c21t::element_p3 p, q;
        
        c21rs::from_bytes(p, commitments_data[0].data());

        c21rs::from_bytes(q, commitments_data[1].data());

        c21o::add(p, p, q);

        c21rs::to_bytes(commitment_c.data(), p);

        sqcb::commitment &expected_commitment_c = commitments_data[2];

        REQUIRE(commitment_c == expected_commitment_c);
    }
}

TEST_CASE("Test 2 - We can add 3 * g as well as g + g + g") {
    const uint64_t num_rows = 1;
    const uint64_t num_columns = 4;
    const uint8_t element_nbytes = sizeof(int);

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_columns);

    const int query[num_columns][num_rows] = {
        {1},// A
        {1},// B
        {1},// C
        {3} // D = A + B + C
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_columns; ++i) {
        sequences[i].n = num_rows;
        sequences[i].data = (const uint8_t *) query[i];
        sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("Verifying if addion property holds (3 = 1 + 1 + 1 ==> commit(3) = commit(1) + commit(1) + commit(1))") {
        sqcnv::compute_commitments_cpu(commitments, value_sequences);
        
        sqcb::commitment commitment_c;

        c21t::element_p3 p, q, s;
        
        c21rs::from_bytes(p, commitments_data[0].data());

        c21rs::from_bytes(q, commitments_data[1].data());

        c21rs::from_bytes(s, commitments_data[2].data());

        c21o::add(p, p, q);

        c21o::add(p, p, s);

        c21rs::to_bytes(commitment_c.data(), p);

        sqcb::commitment &expected_commitment_c = commitments_data[3];

        REQUIRE(commitment_c == expected_commitment_c);
    }
}

TEST_CASE("Test 3 - We can add 3 * g as well as g + g + g by using the add function directly") {
    const uint64_t num_rows = 1;
    const uint64_t num_columns = 1;
    const uint8_t element_nbytes = sizeof(int);

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_columns);

    const int query[num_columns][num_rows] = {
        {3} // D = A + B + C
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_columns; ++i) {
        sequences[i].n = num_rows;
        sequences[i].data = (const uint8_t *) query[i];
        sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("Verifying if addion property holds (commit(3 * g) == commit(g + g + g))") {
        c21t::element_p3 p;
        c21t::element_p3 g_i;

        sqcb::compute_base_element(g_i, 0);

        sqcnv::compute_commitments_cpu(commitments, value_sequences);

        c21o::add(p, g_i, g_i);

        c21o::add(p, p, g_i);

        sqcb::commitment commitment;

        c21rs::to_bytes(commitment.data(), p);

        sqcb::commitment &expected_commitment = commitments_data[0];

        REQUIRE(commitment == expected_commitment);
    }
}

TEST_CASE("Test 4 - We can verify the maximum range allowed by the commitment (regarding the fill_data function)") {
    const uint64_t num_rows = 1;
    const uint64_t num_columns = 3;
    const uint8_t element_nbytes = 32;

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_columns);

    const unsigned char query[3][32] = {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120, },// A
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,}, // B
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,}  // C = A + B
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_columns; ++i) {
        sequences[i].n = num_rows;
        sequences[i].data = (const uint8_t *) query[i];
        sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("Verifying for C = A + B >= p (p = 2^252 + 27742317777372353535851937790883648493)") {
        sqcnv::compute_commitments_cpu(commitments, value_sequences);
        
        sqcb::commitment commitment_c;

        c21t::element_p3 p, q;
        
        c21rs::from_bytes(p, commitments_data[0].data());

        c21rs::from_bytes(q, commitments_data[1].data());

        c21o::add(p, p, q);

        c21rs::to_bytes(commitment_c.data(), p);

        sqcb::commitment &expected_commitment_c = commitments_data[2];

        REQUIRE(commitment_c == expected_commitment_c);
    }
}

TEST_CASE("Test 5 - We can multiply and add two commitments together") {
    const uint64_t num_rows = 4;
    const uint64_t num_columns = 3;
    const uint8_t element_nbytes = sizeof(int);
    const unsigned int multiplicative_constant = 52;

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_columns);

    const int query[num_columns][num_rows] = {
        {       2000,     7500,          5000,      1500},
        {       5000,        0,        400000,        10},
        {
            multiplicative_constant * 2000 + 5000,
            multiplicative_constant * 7500 + 0,
            multiplicative_constant * 5000 + 400000,
            multiplicative_constant * 1500 + 10
        }
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_columns; ++i) {
        sequences[i].n = num_rows;
        sequences[i].data = (const uint8_t *) query[i];
        sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("Verifying if multiplication and addion property holds (c = 52 * a + b ==> commit_c = 52 * commit_a + commit_b)") {
        sqcnv::compute_commitments_cpu(commitments, value_sequences);
        
        sqcb::commitment commitment_c;

        c21t::element_p3 p, q;
        
        c21rs::from_bytes(p, commitments_data[0].data());

        c21rs::from_bytes(q, commitments_data[1].data());

        uint8_t a_i[32];

        sqcnv::fill_data(a_i, (const uint8_t *) &multiplicative_constant, sizeof(multiplicative_constant));

        c21o::scalar_multiply(p, a_i, p); // h_i = a_i * g_i

        c21o::add(p, p, q);

        c21rs::to_bytes(commitment_c.data(), p);

        sqcb::commitment &expected_commitment_c = commitments_data[2];

        REQUIRE(commitment_c == expected_commitment_c);
    }
}

TEST_CASE("Test 6 - We can add two negative values together and generate valid commitments") {
    const uint64_t num_rows = 1;
    const uint64_t num_columns = 3;

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_columns);

    const int result = 256;
    
    const char query[2][num_rows] = {
        {-128}, // (signed binary char 10000000) === (-128 decimal)
        {-128}, // --> (unsigned binary char 10000000) === (128 decimal)
    };

    sequences[0].n = num_rows;
    sequences[0].data = (const uint8_t *) query[0];
    sequences[0].element_nbytes = sizeof(query[0]);

    sequences[1].n = num_rows;
    sequences[1].data = (const uint8_t *) query[1];
    sequences[1].element_nbytes = sizeof(query[1]);

    sequences[2].n = num_rows;
    sequences[2].data = (const uint8_t *) &result;
    sequences[2].element_nbytes = sizeof(result);  

    SECTION("Verifying if addion property holds (-|c| = -|a| + -|b| ==> commit_c = commit_a + commit_b)") {
        sqcnv::compute_commitments_cpu(commitments, value_sequences);
        
        sqcb::commitment commitment_c;

        c21t::element_p3 p, q;
        
        c21rs::from_bytes(p, commitments_data[0].data());

        c21rs::from_bytes(q, commitments_data[1].data());

        c21o::add(p, p, q);

        c21rs::to_bytes(commitment_c.data(), p);

        sqcb::commitment &expected_commitment_c = commitments_data[2];

        REQUIRE(commitment_c == expected_commitment_c);
    }
}

TEST_CASE("Test 7 - We can verify the maximum range allowed by the commitment (regarding the reduction function)") {
    const uint64_t num_rows = 1;
    const uint64_t num_columns = 3;
    const uint8_t element_nbytes = 32;

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_columns);

    // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
    // A = p (decimal)
    // B = p (decimal)
    // C = p + p (decimal)
    std::array<unsigned long long, 4> p = {6346243789798364141ull, 1503914060200516822ull, 0ull, 1152921504606846976ull};
    std::array<unsigned long long, 4> p2 = {12692487579596728282ull, 3007828120401033644ull, 0ull, 2305843009213693952ull};

    const std::array<std::array<unsigned long long, 4>, 3> query = {
        p, // A
        p, // B
        p2  // C = A + B
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_columns; ++i) {
        sequences[i].n = num_rows;
        sequences[i].data = (const uint8_t *) query[i].data();
        sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("Verifying for C = A + B > p (p = 2^252 + 27742317777372353535851937790883648493)") {
        sqcnv::compute_commitments_cpu(commitments, value_sequences);
        
        sqcb::commitment commitment_c;

        c21t::element_p3 p, q;
        
        c21rs::from_bytes(p, commitments_data[0].data());

        c21rs::from_bytes(q, commitments_data[1].data());

        c21o::add(p, p, q);

        c21rs::to_bytes(commitment_c.data(), p);

        sqcb::commitment &expected_commitment_c = commitments_data[2];

        REQUIRE(commitment_c == expected_commitment_c);
    }
}