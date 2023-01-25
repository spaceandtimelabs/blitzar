#include "sxt/seqcommit/naive/commitment_computation_gpu.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/generator/base_element.h"

using namespace sxt;
using namespace sxt::sqcnv;

//--------------------------------------------------------------------------------------------------
// fetch_generators
//--------------------------------------------------------------------------------------------------
static void fetch_generators(basct::span<c21t::element_p3> generators) {
  for (size_t i = 0; i < generators.size(); ++i) {
    sqcgn::compute_base_element(generators[i], i + 3);
  }
}

TEST_CASE("run computation tests") {
  SECTION("we can add two commitments together") {
    const uint64_t num_rows = 4;
    const uint64_t num_sequences = 3;
    const uint8_t element_nbytes = sizeof(int);

    std::vector<c21t::element_p3> generators(num_rows);
    fetch_generators(generators);

    rstt::compressed_element commitments_data[num_sequences];
    mtxb::exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_sequences);

    const int query[num_sequences][num_rows] = {{2000, 7500, 5000, 1500},
                                                {5000, 0, 400000, 10},
                                                {2000 + 5000, 7500 + 0, 5000 + 400000, 1500 + 10}};

    // populating sequence object
    for (uint64_t i = 0; i < num_sequences; ++i) {
      sequences[i].n = num_rows;
      sequences[i].data = reinterpret_cast<const uint8_t*>(query[i]);
      sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("addition property holds (c = a + b ==> commit_c = commit_a "
            "+ commit_b)") {
      compute_commitments_gpu(commitments, value_sequences, generators);

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

  SECTION("We can correctly compute commitments for a sequence with zero length") {
    const uint64_t num_rows = 0;
    const uint64_t num_sequences = 1;
    const uint8_t element_nbytes = sizeof(int);

    std::vector<c21t::element_p3> generators(num_rows);
    fetch_generators(generators);

    rstt::compressed_element commitments_data{123, 2};
    mtxb::exponent_sequence sequences;
    basct::span<rstt::compressed_element> commitments(&commitments_data, num_sequences);
    basct::cspan<mtxb::exponent_sequence> value_sequences(&sequences, num_sequences);

    // populating sequence object
    sequences.n = num_rows;
    sequences.data = nullptr;
    sequences.element_nbytes = element_nbytes;

    compute_commitments_gpu(commitments, value_sequences, generators);

    REQUIRE(sxt::rstt::compressed_element() == commitments_data);
  }

  SECTION("We can add 3 * g as well as g + g + g") {
    const uint64_t num_rows = 1;
    const uint64_t num_sequences = 4;
    const uint8_t element_nbytes = sizeof(int);

    std::vector<c21t::element_p3> generators(num_rows);
    fetch_generators(generators);

    rstt::compressed_element commitments_data[num_sequences];
    mtxb::exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_sequences);

    const int query[num_sequences][num_rows] = {
        {1}, // A
        {1}, // B
        {1}, // C
        {3}  // D = A + B + C
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_sequences; ++i) {
      sequences[i].n = num_rows;
      sequences[i].data = reinterpret_cast<const uint8_t*>(query[i]);
      sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("3 = 1 + 1 + 1 ==> commit(3) = commit(1) + commit(1) + commit(1)") {
      compute_commitments_gpu(commitments, value_sequences, generators);

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

    std::vector<c21t::element_p3> generators(num_rows);
    fetch_generators(generators);

    rstt::compressed_element commitments_data[num_sequences];
    mtxb::exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_sequences);

    const int query[num_sequences][num_rows] = {
        {3} // D = A + B + C
    };

    // populating sequence object
    for (uint64_t i = 0; i < num_sequences; ++i) {
      sequences[i].n = num_rows;
      sequences[i].data = reinterpret_cast<const uint8_t*>(query[i]);
      sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("commit(3 * g) == commit(g + g + g)") {
      c21t::element_p3 p;

      compute_commitments_gpu(commitments, value_sequences, generators);

      c21o::add(p, generators[0], generators[0]);

      c21o::add(p, p, generators[0]);

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

    std::vector<c21t::element_p3> generators(num_rows);
    fetch_generators(generators);

    rstt::compressed_element commitments_data[num_sequences];
    mtxb::exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_sequences);

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
      sequences[i].n = num_rows;
      sequences[i].data = reinterpret_cast<const uint8_t*>(query[i]);
      sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("C = A + B >= p (p = 2^252 + 27742317777372353535851937790883648493)") {
      compute_commitments_gpu(commitments, value_sequences, generators);

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

    std::vector<c21t::element_p3> generators(num_rows);
    fetch_generators(generators);

    rstt::compressed_element commitments_data[num_sequences];
    mtxb::exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_sequences);

    const int query[num_sequences][num_rows] = {
        {2000, 7500, 5000, 1500},
        {5000, 0, 400000, 10},
        {multiplicative_constant * 2000 + 5000, multiplicative_constant * 7500 + 0,
         multiplicative_constant * 5000 + 400000, multiplicative_constant * 1500 + 10}};

    // populating sequence object
    for (uint64_t i = 0; i < num_sequences; ++i) {
      sequences[i].n = num_rows;
      sequences[i].data = reinterpret_cast<const uint8_t*>(query[i]);
      sequences[i].element_nbytes = element_nbytes;
    }

    SECTION("c = 52 * a + b ==> commit_c = 52 * commit_a + commit_b") {
      compute_commitments_gpu(commitments, value_sequences, generators);

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

    std::vector<c21t::element_p3> generators(num_rows);
    fetch_generators(generators);

    rstt::compressed_element commitments_data[num_sequences];
    mtxb::exponent_sequence sequences[num_sequences];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_sequences);

    const int result = 256;

    const char query[2][num_rows] = {
        {-128}, // (signed binary char 10000000) === (-128 decimal)
        {-128}, // --> (unsigned binary char 10000000) === (128 decimal)
    };

    sequences[0].n = num_rows;
    sequences[0].data = reinterpret_cast<const uint8_t*>(query[0]);
    sequences[0].element_nbytes = sizeof(query[0]);

    sequences[1].n = num_rows;
    sequences[1].data = reinterpret_cast<const uint8_t*>(query[1]);
    sequences[1].element_nbytes = sizeof(query[1]);

    sequences[2].n = num_rows;
    sequences[2].data = reinterpret_cast<const uint8_t*>(&result);
    sequences[2].element_nbytes = sizeof(result);

    SECTION("-|c| = -|a| + -|b| ==> commit_c = commit_a + commit_b") {
      compute_commitments_gpu(commitments, value_sequences, generators);

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
    mtxb::exponent_sequence sequences[num_sequences];

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
      sequences[i].n = num_rows;
      sequences[i].data = reinterpret_cast<const uint8_t*>(query[i].data());
      sequences[i].element_nbytes = element_nbytes;
    }

    std::vector<c21t::element_p3> generators(num_rows);
    fetch_generators(generators);

    basct::span<rstt::compressed_element> commitments(commitments_data, num_sequences);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_sequences);

    SECTION("C = A + B > p (p = 2^252 + 27742317777372353535851937790883648493)") {

      compute_commitments_gpu(commitments, value_sequences, generators);

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

    rstt::compressed_element commitments_data[num_commitments];
    mtxb::exponent_sequence sequences[num_commitments];
    basct::span<rstt::compressed_element> commitments(commitments_data, num_commitments);
    basct::cspan<mtxb::exponent_sequence> value_sequences(sequences, num_commitments);

    // A = p (decimal)
    // B = p (decimal)
    // C = p + p (decimal)
    std::array<unsigned long long, 4> a = {6346243789798364141ull, 1503914060200516822ull, 1ull,
                                           1152921504606846976ull};
    std::array<unsigned int, 2> b = {123, 733};
    std::array<unsigned char, 3> c = {121, 200, 135};
    std::array<unsigned long long, 4> d = {6346243789798364385ull, 1503914060200517755ull, 136ull,
                                           1152921504606846976ull};

    std::vector<c21t::element_p3> generators(4);
    fetch_generators(generators);

    // populating sequence object
    // A = a
    sequences[0].n = 4;
    sequences[0].data = reinterpret_cast<const uint8_t*>(a.data());
    sequences[0].element_nbytes = sizeof(unsigned long long);

    // B = b
    sequences[1].n = 2;
    sequences[1].data = reinterpret_cast<const uint8_t*>(b.data());
    sequences[1].element_nbytes = sizeof(unsigned int);

    // C = c
    sequences[2].n = 3;
    sequences[2].data = reinterpret_cast<const uint8_t*>(c.data());
    sequences[2].element_nbytes = sizeof(unsigned char);

    // D = d
    sequences[3].n = 4;
    sequences[3].data = reinterpret_cast<const uint8_t*>(d.data());
    sequences[3].element_nbytes = sizeof(unsigned long long);

    SECTION("D = A + B + C implies in commit(D) = commit(A) + commit(B) + "
            "commit(C)") {
      compute_commitments_gpu(commitments, value_sequences, generators);

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
}
