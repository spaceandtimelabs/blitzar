#include "sxt/seqcommit/naive/commitment_computation.h"

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
    const uint64_t numRows = (uint64_t) 4;
    const uint64_t numColumns = (uint64_t) 3;
    const uint8_t element_nbytes = (uint8_t) sizeof(int);

    sqcb::commitment commitmentsData[numColumns];
    mtxb::exponent_sequence sequences[numColumns];
    basct::span<sqcb::commitment> commitments(commitmentsData, numColumns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, numColumns);

    const int query[numColumns][numRows] = {
        {       2000,     7500,          5000,      1500},
        {       5000,        0,        400000,        10},
        {2000 + 5000, 7500 + 0, 5000 + 400000, 1500 + 10}
    };

    // populating sequence object
    for (uint64_t i = 0; i < numColumns; ++i) {
        sequences[i].n = numRows;
        sequences[i].data = (const uint8_t *) query[i];
        sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("Verifying if addion property holds (c = a + b ==> commit_c = commit_a + commit_b)") {
        sqcnv::compute_commitments(commitments, value_sequences);
        
        sqcb::commitment commitment_c;

        c21t::element_p3 p, q;
        
        c21rs::from_bytes(p, commitmentsData[0].data());

        c21rs::from_bytes(q, commitmentsData[1].data());

        c21o::add(p, p, q);

        c21rs::to_bytes(commitment_c.data(), p);

        sqcb::commitment &expected_commitment_c = commitmentsData[2];

        REQUIRE(commitment_c == expected_commitment_c);
    }
}

TEST_CASE("Test 2 - We can add 3 * g as well as g + g + g") {
    const uint64_t numRows = (uint64_t) 1;
    const uint64_t numColumns = (uint64_t) 4;
    const uint8_t element_nbytes = (uint8_t) sizeof(int);

    sqcb::commitment commitmentsData[numColumns];
    mtxb::exponent_sequence sequences[numColumns];
    basct::span<sqcb::commitment> commitments(commitmentsData, numColumns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, numColumns);

    const int query[numColumns][numRows] = {
        {1},// A
        {1},// B
        {1},// C
        {3} // D = A + B + C
    };

    // populating sequence object
    for (uint64_t i = 0; i < numColumns; ++i) {
        sequences[i].n = numRows;
        sequences[i].data = (const uint8_t *) query[i];
        sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("Verifying if addion property holds (3 = 1 + 1 + 1 ==> commit(3) = commit(1) + commit(1) + commit(1))") {
        sqcnv::compute_commitments(commitments, value_sequences);
        
        sqcb::commitment commitment_c;

        c21t::element_p3 p, q, s;
        
        c21rs::from_bytes(p, commitmentsData[0].data());

        c21rs::from_bytes(q, commitmentsData[1].data());

        c21rs::from_bytes(s, commitmentsData[2].data());

        c21o::add(p, p, q);

        c21o::add(p, p, s);

        c21rs::to_bytes(commitment_c.data(), p);

        sqcb::commitment &expected_commitment_c = commitmentsData[3];

        REQUIRE(commitment_c == expected_commitment_c);
    }
}

TEST_CASE("Test 3 - We can add 3 * g as well as g + g + g by using the add function directly") {
    const uint64_t numRows = (uint64_t) 1;
    const uint64_t numColumns = (uint64_t) 1;
    const uint8_t element_nbytes = (uint8_t) sizeof(int);

    sqcb::commitment commitmentsData[numColumns];
    mtxb::exponent_sequence sequences[numColumns];
    basct::span<sqcb::commitment> commitments(commitmentsData, numColumns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, numColumns);

    const int query[numColumns][numRows] = {
        {3} // D = A + B + C
    };

    // populating sequence object
    for (uint64_t i = 0; i < numColumns; ++i) {
        sequences[i].n = numRows;
        sequences[i].data = (const uint8_t *) query[i];
        sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("Verifying if addion property holds (commit(3 * g) == commit(g + g + g))") {
        c21t::element_p3 p;
        c21t::element_p3 g_i;

        sqcb::compute_base_element(g_i, 0);

        sqcnv::compute_commitments(commitments, value_sequences);

        c21o::add(p, g_i, g_i);

        c21o::add(p, p, g_i);

        sqcb::commitment commitment;

        c21rs::to_bytes(commitment.data(), p);

        sqcb::commitment &expected_commitment = commitmentsData[0];

        REQUIRE(commitment == expected_commitment);
    }
}

TEST_CASE("Test 4 - We can verify the maximum range allowed by the commitment") {
    const uint64_t numRows = (uint64_t) 1;
    const uint64_t numColumns = (uint64_t) 3;
    const uint8_t element_nbytes = (uint8_t) 32;

    sqcb::commitment commitmentsData[numColumns];
    mtxb::exponent_sequence sequences[numColumns];
    basct::span<sqcb::commitment> commitments(commitmentsData, numColumns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, numColumns);

    const unsigned char query[3][32] = {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120, }, // A
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,}, // B
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,} // C = A + B
    };

    // populating sequence object
    for (uint64_t i = 0; i < numColumns; ++i) {
        sequences[i].n = numRows;
        sequences[i].data = (const uint8_t *) query[i];
        sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("Verifying for C = A + B >= p (p = 2^252 + 27742317777372353535851937790883648493)") {
        sqcnv::compute_commitments(commitments, value_sequences);
        
        sqcb::commitment commitment_c;

        c21t::element_p3 p, q;
        
        c21rs::from_bytes(p, commitmentsData[0].data());

        c21rs::from_bytes(q, commitmentsData[1].data());

        c21o::add(p, p, q);

        c21rs::to_bytes(commitment_c.data(), p);

        sqcb::commitment &expected_commitment_c = commitmentsData[2];

        REQUIRE(commitment_c == expected_commitment_c);
    }
}

TEST_CASE("Test 5 - We can multiply and add two commitments together") {
    const uint64_t numRows = (uint64_t) 4;
    const uint64_t numColumns = (uint64_t) 3;
    const unsigned int multiplicative_constant = 52;
    const uint8_t element_nbytes = (uint8_t) sizeof(int);

    sqcb::commitment commitmentsData[numColumns];
    mtxb::exponent_sequence sequences[numColumns];
    basct::span<sqcb::commitment> commitments(commitmentsData, numColumns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, numColumns);

    const int query[numColumns][numRows] = {
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
    for (uint64_t i = 0; i < numColumns; ++i) {
        sequences[i].n = numRows;
        sequences[i].data = (const uint8_t *) query[i];
        sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("Verifying if multiplication and addion property holds (c = 52 * a + b ==> commit_c = 52 * commit_a + commit_b)") {
        sqcnv::compute_commitments(commitments, value_sequences);
        
        sqcb::commitment commitment_c;

        c21t::element_p3 p, q;
        
        c21rs::from_bytes(p, commitmentsData[0].data());

        c21rs::from_bytes(q, commitmentsData[1].data());

        uint8_t a_i[32];

        sqcnv::fill_data(a_i, (const uint8_t *) &multiplicative_constant, sizeof(multiplicative_constant));

        c21o::scalar_multiply(p, a_i, p); // h_i = a_i * g_i

        c21o::add(p, p, q);

        c21rs::to_bytes(commitment_c.data(), p);

        sqcb::commitment &expected_commitment_c = commitmentsData[2];

        REQUIRE(commitment_c == expected_commitment_c);
    }
}

TEST_CASE("Test 6 - We can add two negative values together and generate valid commitments") {
    const uint64_t numRows = (uint64_t) 1;
    const uint64_t numColumns = (uint64_t) 3;

    sqcb::commitment commitmentsData[numColumns];
    mtxb::exponent_sequence sequences[numColumns];
    basct::span<sqcb::commitment> commitments(commitmentsData, numColumns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, numColumns);

    const char query[2][numRows] = {
        {-128}, // (signed binary char 10000000) === (-128 decimal)
        {-128}, // --> (unsigned binary char 10000000) === (128 decimal)
    };

    int result = 256;

    sequences[0].n = numRows;
    sequences[0].data = (const uint8_t *) query[0];
    sequences[0].element_nbytes = sizeof(query[0]);

    sequences[1].n = numRows;
    sequences[1].data = (const uint8_t *) query[1];
    sequences[1].element_nbytes = sizeof(query[1]);

    sequences[2].n = numRows;
    sequences[2].data = (const uint8_t *) &result;
    sequences[2].element_nbytes = sizeof(result);  

    SECTION("Verifying if addion property holds (-|c| = -|a| + -|b| ==> commit_c = commit_a + commit_b)") {
        sqcnv::compute_commitments(commitments, value_sequences);
        
        sqcb::commitment commitment_c;

        c21t::element_p3 p, q;
        
        c21rs::from_bytes(p, commitmentsData[0].data());

        c21rs::from_bytes(q, commitmentsData[1].data());

        c21o::add(p, p, q);

        c21rs::to_bytes(commitment_c.data(), p);

        sqcb::commitment &expected_commitment_c = commitmentsData[2];

        REQUIRE(commitment_c == expected_commitment_c);
    }
}