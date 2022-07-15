#include "sxt/seqcommit/test/test_pedersen.h"

#include "sxt/base/test/unit_test.h"

#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/type/element_p3.h"

#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

#include "sxt/seqcommit/base/indexed_exponent_sequence.h"
#include "sxt/seqcommit/generator/base_element.h"

namespace sxt::sqctst {
//--------------------------------------------------------------------------------------------------
// test_pedersen_compute_commitment
//--------------------------------------------------------------------------------------------------
void test_pedersen_compute_commitment(
    basf::function_ref<void(basct::span<rstt::compressed_element>,
                            basct::cspan<sqcb::indexed_exponent_sequence>,
                            basct::cspan<rstt::compressed_element> generators)>
        f) {
  SECTION("we can add two commitments together") {
    const uint64_t num_rows = 4;
    const uint64_t num_sequences = 3;
    const uint8_t element_nbytes = sizeof(int);

    basct::span<rstt::compressed_element> empty_generators;
    rstt::compressed_element commitments_data[num_sequences];
    sqcb::indexed_exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences, num_sequences);

    const int query[num_sequences][num_rows] = {{2000, 7500, 5000, 1500},
                                                {5000, 0, 400000, 10},
                                                {2000 + 5000, 7500 + 0, 5000 + 400000, 1500 + 10}};

    // populating sequence object
    for (uint64_t i = 0; i < num_sequences; ++i) {
      sequences[i].exponent_sequence.n = num_rows;
      sequences[i].exponent_sequence.data = reinterpret_cast<const uint8_t*>(query[i]);
      sequences[i].exponent_sequence.element_nbytes = element_nbytes;
    }

    SECTION("addition property holds (c = a + b ==> commit_c = commit_a "
            "+ commit_b)") {
      f(commitments, value_sequences, empty_generators);

      rstt::compressed_element commitment_c;

      c21t::element_p3 p, q;

      rstb::from_bytes(p, commitments_data[0].data());

      rstb::from_bytes(q, commitments_data[1].data());

      c21o::add(p, p, q);

      rstb::to_bytes(commitment_c.data(), p);

      rstt::compressed_element& expected_commitment_c = commitments_data[2];

      // verify that result is not null
      REQUIRE(sxt::rstt::compressed_element() != commitment_c);

      REQUIRE(commitment_c == expected_commitment_c);
    }
  }

  SECTION("We can add 3 * g as well as g + g + g") {
    const uint64_t num_rows = 1;
    const uint64_t num_sequences = 4;
    const uint8_t element_nbytes = sizeof(int);

    basct::span<rstt::compressed_element> empty_generators;
    rstt::compressed_element commitments_data[num_sequences];
    sqcb::indexed_exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences, num_sequences);

    const int query[num_sequences][num_rows] = {
        {1}, // A
        {1}, // B
        {1}, // C
        {3}  // D = A + B + C
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_sequences; ++i) {
      sequences[i].exponent_sequence.n = num_rows;
      sequences[i].exponent_sequence.data = reinterpret_cast<const uint8_t*>(query[i]);
      sequences[i].exponent_sequence.element_nbytes = element_nbytes;
    }

    SECTION("3 = 1 + 1 + 1 ==> commit(3) = commit(1) + commit(1) + commit(1)") {
      f(commitments, value_sequences, empty_generators);

      rstt::compressed_element commitment_c;

      c21t::element_p3 p, q, s;

      rstb::from_bytes(p, commitments_data[0].data());

      rstb::from_bytes(q, commitments_data[1].data());

      rstb::from_bytes(s, commitments_data[2].data());

      c21o::add(p, p, q);

      c21o::add(p, p, s);

      rstb::to_bytes(commitment_c.data(), p);

      rstt::compressed_element& expected_commitment_c = commitments_data[3];

      // verify that result is not null
      REQUIRE(sxt::rstt::compressed_element() != commitment_c);

      REQUIRE(commitment_c == expected_commitment_c);
    }
  }

  SECTION("We can add 3 * g as well as g + g + g by using the add function "
          "directly") {
    const uint64_t num_rows = 1;
    const uint64_t num_sequences = 1;
    const uint8_t element_nbytes = sizeof(int);

    basct::span<rstt::compressed_element> empty_generators;
    rstt::compressed_element commitments_data[num_sequences];
    sqcb::indexed_exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences, num_sequences);

    const int query[num_sequences][num_rows] = {
        {3} // D = A + B + C
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_sequences; ++i) {
      sequences[i].exponent_sequence.n = num_rows;
      sequences[i].exponent_sequence.data = reinterpret_cast<const uint8_t*>(query[i]);
      sequences[i].exponent_sequence.element_nbytes = element_nbytes;
    }

    SECTION("commit(3 * g) == commit(g + g + g)") {
      c21t::element_p3 p;
      c21t::element_p3 g_i;

      sqcgn::compute_base_element(g_i, 0);

      f(commitments, value_sequences, empty_generators);

      c21o::add(p, g_i, g_i);

      c21o::add(p, p, g_i);

      rstt::compressed_element commitment;

      rstb::to_bytes(commitment.data(), p);

      rstt::compressed_element& expected_commitment = commitments_data[0];

      // verify that result is not null
      REQUIRE(sxt::rstt::compressed_element() != commitment);

      REQUIRE(commitment == expected_commitment);
    }
  }

  SECTION("We can verify the maximum range allowed by the commitment (regarding "
          "the fill_data function)") {
    const uint64_t num_rows = 1;
    const uint64_t num_sequences = 3;
    const uint8_t element_nbytes = 32;

    basct::span<rstt::compressed_element> empty_generators;
    rstt::compressed_element commitments_data[num_sequences];
    sqcb::indexed_exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences, num_sequences);

    const unsigned char query[3][32] = {
        {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120,
        }, // A
        {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120,
        }, // B
        {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240,
        } // C = A + B
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_sequences; ++i) {
      sequences[i].exponent_sequence.n = num_rows;
      sequences[i].exponent_sequence.data = reinterpret_cast<const uint8_t*>(query[i]);
      sequences[i].exponent_sequence.element_nbytes = element_nbytes;
    }

    SECTION("C = A + B >= p (p = 2^252 + 27742317777372353535851937790883648493)") {
      f(commitments, value_sequences, empty_generators);

      rstt::compressed_element commitment_c;

      c21t::element_p3 p, q;

      rstb::from_bytes(p, commitments_data[0].data());

      rstb::from_bytes(q, commitments_data[1].data());

      c21o::add(p, p, q);

      rstb::to_bytes(commitment_c.data(), p);

      rstt::compressed_element& expected_commitment_c = commitments_data[2];

      // verify that result is not null
      REQUIRE(sxt::rstt::compressed_element() != commitment_c);

      REQUIRE(commitment_c == expected_commitment_c);
    }
  }

  SECTION("We can multiply and add two commitments together") {
    const uint64_t num_rows = 4;
    const uint64_t num_sequences = 3;
    const uint8_t element_nbytes = sizeof(int);
    const unsigned int multiplicative_constant = 52;

    basct::span<rstt::compressed_element> empty_generators;
    rstt::compressed_element commitments_data[num_sequences];
    sqcb::indexed_exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences, num_sequences);

    const int query[num_sequences][num_rows] = {
        {2000, 7500, 5000, 1500},
        {5000, 0, 400000, 10},
        {multiplicative_constant * 2000 + 5000, multiplicative_constant * 7500 + 0,
         multiplicative_constant * 5000 + 400000, multiplicative_constant * 1500 + 10}};

    // populating sequence object
    for (uint64_t i = 0; i < num_sequences; ++i) {
      sequences[i].exponent_sequence.n = num_rows;
      sequences[i].exponent_sequence.data = reinterpret_cast<const uint8_t*>(query[i]);
      sequences[i].exponent_sequence.element_nbytes = element_nbytes;
    }

    SECTION("c = 52 * a + b ==> commit_c = 52 * commit_a + commit_b") {
      f(commitments, value_sequences, empty_generators);

      rstt::compressed_element commitment_c;

      c21t::element_p3 p, q;

      rstb::from_bytes(p, commitments_data[0].data());

      rstb::from_bytes(q, commitments_data[1].data());

      c21o::scalar_multiply(p, multiplicative_constant,
                            p); // h_i = a_i * g_i

      c21o::add(p, p, q);

      rstb::to_bytes(commitment_c.data(), p);

      rstt::compressed_element& expected_commitment_c = commitments_data[2];

      // verify that result is not null
      REQUIRE(sxt::rstt::compressed_element() != commitment_c);

      REQUIRE(commitment_c == expected_commitment_c);
    }
  }

  SECTION("We can add two negative values together and generate valid "
          "commitments") {
    const uint64_t num_rows = 1;
    const uint64_t num_sequences = 3;

    basct::span<rstt::compressed_element> empty_generators;
    rstt::compressed_element commitments_data[num_sequences];
    sqcb::indexed_exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences, num_sequences);

    const int result = 256;

    const char query[2][num_rows] = {
        {-128}, // (signed binary char 10000000) === (-128 decimal)
        {-128}, // --> (unsigned binary char 10000000) === (128 decimal)
    };

    sequences[0].exponent_sequence.n = num_rows;
    sequences[0].exponent_sequence.data = reinterpret_cast<const uint8_t*>(query[0]);
    sequences[0].exponent_sequence.element_nbytes = sizeof(query[0]);

    sequences[1].exponent_sequence.n = num_rows;
    sequences[1].exponent_sequence.data = reinterpret_cast<const uint8_t*>(query[1]);
    sequences[1].exponent_sequence.element_nbytes = sizeof(query[1]);

    sequences[2].exponent_sequence.n = num_rows;
    sequences[2].exponent_sequence.data = reinterpret_cast<const uint8_t*>(&result);
    sequences[2].exponent_sequence.element_nbytes = sizeof(result);

    SECTION("-|c| = -|a| + -|b| ==> commit_c = commit_a + commit_b") {
      f(commitments, value_sequences, empty_generators);

      rstt::compressed_element commitment_c;

      c21t::element_p3 p, q;

      rstb::from_bytes(p, commitments_data[0].data());

      rstb::from_bytes(q, commitments_data[1].data());

      c21o::add(p, p, q);

      rstb::to_bytes(commitment_c.data(), p);

      rstt::compressed_element& expected_commitment_c = commitments_data[2];

      // verify that result is not null
      REQUIRE(sxt::rstt::compressed_element() != commitment_c);

      REQUIRE(commitment_c == expected_commitment_c);
    }
  }

  SECTION("We can verify the maximum range allowed by the commitment "
          "(regarding the reduction function)") {
    const uint64_t num_rows = 1;
    const uint64_t num_sequences = 3;
    const uint8_t element_nbytes = 32;

    rstt::compressed_element commitments_data[num_sequences];
    sqcb::indexed_exponent_sequence sequences[num_sequences];

    // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
    // A = p (decimal)
    // B = p (decimal)
    // C = p + p (decimal)
    std::array<unsigned long long, 4> pval = {6346243789798364141ull, 1503914060200516822ull, 0ull,
                                              1152921504606846976ull};
    std::array<unsigned long long, 4> p2val = {12692487579596728282ull, 3007828120401033644ull,
                                               0ull, 2305843009213693952ull};

    const std::array<std::array<unsigned long long, 4>, 3> query = {
        pval, // A
        pval, // B
        p2val // C = A + B
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_sequences; ++i) {
      sequences[i].exponent_sequence.n = num_rows;
      sequences[i].exponent_sequence.data = reinterpret_cast<const uint8_t*>(query[i].data());
      sequences[i].exponent_sequence.element_nbytes = element_nbytes;
    }

    basct::span<rstt::compressed_element> empty_generators(nullptr, 0);
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences, num_sequences);

    SECTION("C = A + B > p (p = 2^252 + 27742317777372353535851937790883648493)") {

      f(commitments, value_sequences, empty_generators);

      rstt::compressed_element commitment_c;

      c21t::element_p3 p, q;

      rstb::from_bytes(p, commitments_data[0].data());

      rstb::from_bytes(q, commitments_data[1].data());

      c21o::add(p, p, q);

      rstb::to_bytes(commitment_c.data(), p);

      rstt::compressed_element& expected_commitment_c = commitments_data[2];

      // verify that result is not null
      REQUIRE(sxt::rstt::compressed_element() == commitment_c);

      REQUIRE(commitment_c == expected_commitment_c);
    }
  }

  SECTION("We can verify that adding columns with different word size and "
          "number of rows") {
    const uint64_t num_commitments = 4;

    basct::span<rstt::compressed_element> empty_generators;
    rstt::compressed_element commitments_data[num_commitments];
    sqcb::indexed_exponent_sequence sequences[num_commitments];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_commitments);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences, num_commitments);

    // A = p (decimal)
    // B = p (decimal)
    // C = p + p (decimal)
    std::array<unsigned long long, 4> a = {6346243789798364141ull, 1503914060200516822ull, 1ull,
                                           1152921504606846976ull};
    std::array<unsigned int, 2> b = {123, 733};
    std::array<unsigned char, 3> c = {121, 200, 135};
    std::array<unsigned long long, 4> d = {6346243789798364385ull, 1503914060200517755ull, 136ull,
                                           1152921504606846976ull};

    // populating sequence object
    // A = a
    sequences[0].exponent_sequence.n = 4;
    sequences[0].exponent_sequence.data = reinterpret_cast<const uint8_t*>(a.data());
    sequences[0].exponent_sequence.element_nbytes = sizeof(unsigned long long);

    // B = b
    sequences[1].exponent_sequence.n = 2;
    sequences[1].exponent_sequence.data = reinterpret_cast<const uint8_t*>(b.data());
    sequences[1].exponent_sequence.element_nbytes = sizeof(unsigned int);

    // C = c
    sequences[2].exponent_sequence.n = 3;
    sequences[2].exponent_sequence.data = reinterpret_cast<const uint8_t*>(c.data());
    sequences[2].exponent_sequence.element_nbytes = sizeof(unsigned char);

    // D = d
    sequences[3].exponent_sequence.n = 4;
    sequences[3].exponent_sequence.data = reinterpret_cast<const uint8_t*>(d.data());
    sequences[3].exponent_sequence.element_nbytes = sizeof(unsigned long long);

    SECTION("D = A + B + C implies in commit(D) = commit(A) + commit(B) + "
            "commit(C)") {
      f(commitments, value_sequences, empty_generators);

      rstt::compressed_element commitment_c;

      c21t::element_p3 p, q, r;

      rstb::from_bytes(p, commitments_data[0].data());

      rstb::from_bytes(q, commitments_data[1].data());

      rstb::from_bytes(r, commitments_data[2].data());

      c21o::add(p, p, q);

      c21o::add(p, p, r);

      rstb::to_bytes(commitment_c.data(), p);

      rstt::compressed_element& expected_commitment_c = commitments_data[3];

      // verify that result is not null
      REQUIRE(sxt::rstt::compressed_element() != commitment_c);

      REQUIRE(commitment_c == expected_commitment_c);
    }
  }

  SECTION("We can pass generators to the commitment computation") {

    rstt::compressed_element ristretto_gens[4];
    const int data_values[4] = {1, 2, 3, 4};
    c21t::element_p3 expected_g = c21cn::zero_p3_v;

    for (int i = 0; i < 4; ++i) {
      c21t::element_p3 g_i;
      sqcgn::compute_base_element(g_i, i + 3);

      rstb::to_bytes(ristretto_gens[i].data(), g_i);
      rstb::from_bytes(g_i, ristretto_gens[i].data());

      c21t::element_p3 h;
      c21o::scalar_multiply(
          h, basct::cspan<uint8_t>{reinterpret_cast<const uint8_t*>(data_values + i), sizeof(int)},
          g_i);

      c21o::add(expected_g, expected_g, h);
    }

    rstt::compressed_element expected_commitment;
    rstb::to_bytes(expected_commitment.data(), expected_g);

    rstt::compressed_element commitments_data[1];
    sqcb::indexed_exponent_sequence sequences[1];

    sequences[0].exponent_sequence.n = 4;
    sequences[0].exponent_sequence.data = reinterpret_cast<const uint8_t*>(data_values);
    sequences[0].exponent_sequence.element_nbytes = sizeof(int);

    basct::span<rstt::compressed_element> commitments(commitments_data, 1);
    basct::span<rstt::compressed_element> span_generators(ristretto_gens, 4);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences, 1);

    f(commitments, value_sequences, span_generators);

    // verify that result is not null
    REQUIRE(sxt::rstt::compressed_element() != commitments_data[0]);

    REQUIRE(expected_commitment == commitments_data[0]);
  }

  SECTION("We can compute sparse commitments") {
    const uint64_t num_sequences = 2;
    const uint8_t element_nbytes = sizeof(int);

    sxt::rstt::compressed_element commitments_data[num_sequences];

    const int dense_data[11] = {1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0};

    const int sparse_data[5] = {1, 2, 3, 4, 9};

    const uint64_t sparse_indices[5] = {0, 2, 4, 5, 9};

    sqcb::indexed_exponent_sequence dense_descriptor = {
        element_nbytes, 11, reinterpret_cast<const uint8_t*>(dense_data), nullptr};

    sqcb::indexed_exponent_sequence sparse_descriptor = {
        element_nbytes, 5, reinterpret_cast<const uint8_t*>(sparse_data), sparse_indices};

    sqcb::indexed_exponent_sequence sequences[num_sequences] = {dense_descriptor,
                                                                sparse_descriptor};

    basct::span<rstt::compressed_element> span_generators(nullptr, 0);
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences, num_sequences);

    SECTION("sparse_commitments == dense_commitment") {
      f(commitments, value_sequences, span_generators);

      // verify that result is not null
      REQUIRE(sxt::rstt::compressed_element() != commitments_data[0]);

      REQUIRE(commitments_data[0] == commitments_data[1]);
    }
  }

  SECTION("We can compute dense commitments as sparse commitments") {
    const uint64_t num_sequences = 2;
    const uint8_t element_nbytes = sizeof(int);

    rstt::compressed_element commitments_data[num_sequences];

    const int dense_data[11] = {1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0};

    const int sparse_data[11] = {1, 0, 2, 0, 3, 4, 0, 0, 0, 9, 0};

    const uint64_t sparse_indices[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    sqcb::indexed_exponent_sequence dense_descriptor = {
        element_nbytes, 11, reinterpret_cast<const uint8_t*>(dense_data), nullptr};

    sqcb::indexed_exponent_sequence sparse_descriptor = {
        element_nbytes, 11, reinterpret_cast<const uint8_t*>(sparse_data), sparse_indices};

    sqcb::indexed_exponent_sequence sequences[num_sequences] = {dense_descriptor,
                                                                sparse_descriptor};

    basct::span<rstt::compressed_element> span_generators(nullptr, 0);
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences(sequences, num_sequences);

    SECTION("sparse_commitments == dense_commitment") {
      f(commitments, value_sequences, span_generators);

      // verify that result is not null
      REQUIRE(sxt::rstt::compressed_element() != commitments_data[0]);

      REQUIRE(commitments_data[0] == commitments_data[1]);
    }
  }
}

} // namespace sxt::sqctst
