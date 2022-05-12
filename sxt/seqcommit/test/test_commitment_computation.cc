#include "sxt/seqcommit/test/test_commitment_computation.h"

#include "sxt/curve21/type/element_p3.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/curve21/ristretto/byte_conversion.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/seqcommit/base/base_element.h"
#include "sxt/curve21/operation/scalar_multiply.h"

namespace sxt::sqctst {
//--------------------------------------------------------------------------------------------------
// test_commitment_computation_function
//--------------------------------------------------------------------------------------------------
void test_commitment_computation_function(
    basf::function_ref<void(basct::span<sqcb::commitment>,
                            basct::cspan<mtxb::exponent_sequence>)>
        f) {
  SECTION("we can add two commitments together") {
    const uint64_t num_rows = 4;
    const uint64_t num_columns = 3;
    const uint8_t element_nbytes = sizeof(int);

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences,
                                                          num_columns);

    const int query[num_columns][num_rows] = {
        {2000, 7500, 5000, 1500},
        {5000, 0, 400000, 10},
        {2000 + 5000, 7500 + 0, 5000 + 400000, 1500 + 10}};

    // populating sequence object
    for (uint64_t i = 0; i < num_columns; ++i) {
      sequences[i].n = num_rows;
      sequences[i].data = (const uint8_t *)query[i];
      sequences[i].element_nbytes = element_nbytes;
    }

    SECTION(
        "addition property holds (c = a + b ==> commit_c = commit_a "
        "+ commit_b)") {
      f(commitments, value_sequences);

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

  SECTION("We can add 3 * g as well as g + g + g") {
    const uint64_t num_rows = 1;
    const uint64_t num_columns = 4;
    const uint8_t element_nbytes = sizeof(int);

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences,
                                                          num_columns);

    const int query[num_columns][num_rows] = {
        {1},  // A
        {1},  // B
        {1},  // C
        {3}   // D = A + B + C
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_columns; ++i) {
      sequences[i].n = num_rows;
      sequences[i].data = (const uint8_t *)query[i];
      sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("3 = 1 + 1 + 1 ==> commit(3) = commit(1) + commit(1) + commit(1)") {
      f(commitments, value_sequences);

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

  SECTION(
      "We can add 3 * g as well as g + g + g by using the add function "
      "directly") {
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

    SECTION("commit(3 * g) == commit(g + g + g)") {
        c21t::element_p3 p;
        c21t::element_p3 g_i;

        sqcb::compute_base_element(g_i, 0);

        f(commitments, value_sequences);

        c21o::add(p, g_i, g_i);

        c21o::add(p, p, g_i);

        sqcb::commitment commitment;

        c21rs::to_bytes(commitment.data(), p);

        sqcb::commitment &expected_commitment = commitments_data[0];

        REQUIRE(commitment == expected_commitment);
    }
  }

  SECTION(
      "We can verify the maximum range allowed by the commitment (regarding "
      "the fill_data function)") {
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

    SECTION(
        "C = A + B >= p (p = 2^252 + 27742317777372353535851937790883648493)") {
      f(commitments, value_sequences);

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

  SECTION("We can multiply and add two commitments together") {
    const uint64_t num_rows = 4;
    const uint64_t num_columns = 3;
    const uint8_t element_nbytes = sizeof(int);
    const unsigned int multiplicative_constant = 52;

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences,
                                                          num_columns);

    const int query[num_columns][num_rows] = {
        {2000, 7500, 5000, 1500},
        {5000, 0, 400000, 10},
        {multiplicative_constant * 2000 + 5000,
         multiplicative_constant * 7500 + 0,
         multiplicative_constant * 5000 + 400000,
         multiplicative_constant * 1500 + 10}};

    // populating sequence object
    for (uint64_t i = 0; i < num_columns; ++i) {
      sequences[i].n = num_rows;
      sequences[i].data = (const uint8_t *)query[i];
      sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("c = 52 * a + b ==> commit_c = 52 * commit_a + commit_b") {
      f(commitments, value_sequences);

      sqcb::commitment commitment_c;

      c21t::element_p3 p, q;

      c21rs::from_bytes(p, commitments_data[0].data());

      c21rs::from_bytes(q, commitments_data[1].data());

      c21o::scalar_multiply(p, multiplicative_constant,
                            p);  // h_i = a_i * g_i

      c21o::add(p, p, q);

      c21rs::to_bytes(commitment_c.data(), p);

      sqcb::commitment &expected_commitment_c = commitments_data[2];

      REQUIRE(commitment_c == expected_commitment_c);
    }
  }

  SECTION(
      "We can add two negative values together and generate valid "
      "commitments") {
    const uint64_t num_rows = 1;
    const uint64_t num_columns = 3;

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences,
                                                          num_columns);

    const int result = 256;

    const char query[2][num_rows] = {
        {-128},  // (signed binary char 10000000) === (-128 decimal)
        {-128},  // --> (unsigned binary char 10000000) === (128 decimal)
    };

    sequences[0].n = num_rows;
    sequences[0].data = (const uint8_t *)query[0];
    sequences[0].element_nbytes = sizeof(query[0]);

    sequences[1].n = num_rows;
    sequences[1].data = (const uint8_t *)query[1];
    sequences[1].element_nbytes = sizeof(query[1]);

    sequences[2].n = num_rows;
    sequences[2].data = (const uint8_t *)&result;
    sequences[2].element_nbytes = sizeof(result);

    SECTION("-|c| = -|a| + -|b| ==> commit_c = commit_a + commit_b") {
      f(commitments, value_sequences);

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

  SECTION(
      "We can verify the maximum range allowed by the commitment "
      "(regarding the reduction function)") {
    const uint64_t num_rows = 1;
    const uint64_t num_columns = 3;
    const uint8_t element_nbytes = 32;

    sqcb::commitment commitments_data[num_columns];
    mtxb::exponent_sequence sequences[num_columns];
    basct::span<sqcb::commitment> commitments(commitments_data, num_columns);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences,
                                                          num_columns);

    // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
    // A = p (decimal)
    // B = p (decimal)
    // C = p + p (decimal)
    std::array<unsigned long long, 4> p = {6346243789798364141ull,
                                           1503914060200516822ull, 0ull,
                                           1152921504606846976ull};
    std::array<unsigned long long, 4> p2 = {12692487579596728282ull,
                                            3007828120401033644ull, 0ull,
                                            2305843009213693952ull};

    const std::array<std::array<unsigned long long, 4>, 3> query = {
        p,  // A
        p,  // B
        p2  // C = A + B
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_columns; ++i) {
      sequences[i].n = num_rows;
      sequences[i].data = (const uint8_t *)query[i].data();
      sequences[i].element_nbytes = element_nbytes;
    }

    SECTION(
        "C = A + B > p (p = 2^252 + 27742317777372353535851937790883648493)") {
      f(commitments, value_sequences);

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

  SECTION(
      "We can verify that adding columns with different word size and "
      "number of rows") {
    const uint64_t num_commitments = 4;

    sqcb::commitment commitments_data[num_commitments];
    mtxb::exponent_sequence sequences[num_commitments];
    basct::span<sqcb::commitment> commitments(commitments_data,
                                              num_commitments);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences,
                                                          num_commitments);

    // A = p (decimal)
    // B = p (decimal)
    // C = p + p (decimal)
    std::array<unsigned long long, 4> a = {6346243789798364141ull,
                                           1503914060200516822ull, 1ull,
                                           1152921504606846976ull};
    std::array<unsigned int, 2> b = {123, 733};
    std::array<unsigned char, 3> c = {121, 200, 135};
    std::array<unsigned long long, 4> d = {6346243789798364385ull,
                                           1503914060200517755ull, 136ull,
                                           1152921504606846976ull};

    // populating sequence object
    // A = a
    sequences[0].n = 4;
    sequences[0].data = (const uint8_t *)a.data();
    sequences[0].element_nbytes = sizeof(unsigned long long);

    // B = b
    sequences[1].n = 2;
    sequences[1].data = (const uint8_t *)b.data();
    sequences[1].element_nbytes = sizeof(unsigned int);

    // C = c
    sequences[2].n = 3;
    sequences[2].data = (const uint8_t *)c.data();
    sequences[2].element_nbytes = sizeof(unsigned char);

    // D = d
    sequences[3].n = 4;
    sequences[3].data = (const uint8_t *)d.data();
    sequences[3].element_nbytes = sizeof(unsigned long long);

    SECTION(
        "D = A + B + C implies in commit(D) = commit(A) + commit(B) + "
        "commit(C)") {
      f(commitments, value_sequences);

      sqcb::commitment commitment_c;

      c21t::element_p3 p, q, r;

      c21rs::from_bytes(p, commitments_data[0].data());

      c21rs::from_bytes(q, commitments_data[1].data());

      c21rs::from_bytes(r, commitments_data[2].data());

      c21o::add(p, p, q);

      c21o::add(p, p, r);

      c21rs::to_bytes(commitment_c.data(), p);

      sqcb::commitment &expected_commitment_c = commitments_data[3];

      REQUIRE(commitment_c == expected_commitment_c);
    }
  }
}
}  // namespace sxt::sqctst
