#include "sxt/seqcommit/naive/commitment_computation.h"

#include "sxt/curve21/type/element_p3.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/curve21/ristretto/byte_conversion.h"
#include "sxt/curve21/operation/add.h"

using namespace sxt;
using namespace sxt::sqcnv;

TEST_CASE("") {
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
        sequences[i].data = (const void*) query[i];
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
