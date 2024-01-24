/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sxt/multiexp/test/multiexponentiation.h"

#include <cstdint>
#include <limits>
#include <memory_resource>
#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/overload.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/base/exponent_sequence_utility.h"
#include "sxt/multiexp/random/random_multiexponentiation_descriptor.h"
#include "sxt/multiexp/random/random_multiexponentiation_generation.h"
#include "sxt/multiexp/test/curve21_arithmetic.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/literal.h"

using sxt::rstt::operator""_rs;

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// exercise_multiexponentiation_fn
//--------------------------------------------------------------------------------------------------
void exercise_multiexponentiation_fn(std::mt19937& rng, multiexponentiation_fn f) noexcept {
  // bespoke test cases
  SECTION("we handle the empty case") {
    std::vector<mtxb::exponent_sequence> sequences;
    auto res = f({}, {});
    REQUIRE(res.empty());
  }

  SECTION("we handle a sequence with no exponents") {
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = 0, .data = nullptr}};
    auto res = f({}, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        c21t::element_p3::identity(),
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle an exponent of zero") {
    std::vector<uint32_t> exponents = {0};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = f(generators, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        c21t::element_p3::identity(),
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle an exponent of one") {
    std::vector<uint32_t> exponents = {1};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = f(generators, sequences);
    REQUIRE(res == generators);
  }

  SECTION("we handle an exponent of negative one") {
    std::vector<int32_t> exponents = {-1};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = f(generators, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        -generators[0],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle an exponent of two") {
    std::vector<uint32_t> exponents = {2};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = f(generators, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        generators[0] + generators[0],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle an exponent of three") {
    std::vector<uint32_t> exponents = {3};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = f(generators, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        generators[0] + generators[0] + generators[0],
    };
    REQUIRE(res == expected);
  }

  SECTION("we can handle a large exponent") {
    std::vector<uint64_t> exponents = {std::numeric_limits<uint64_t>::max()};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = f(generators, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        exponents[0] * generators[0],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle sequences with more than a single element") {
    std::vector<uint16_t> exponents = {3, 2};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
        0x456_rs,
    };
    auto res = f(generators, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        3 * generators[0] + 2 * generators[1],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle sequences with a mix of positive and negative elements") {
    std::vector<int> exponents = {3, -2, 4, -1};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
        0x456_rs,
        0x789_rs,
        0x101_rs,
    };
    auto res = f(generators, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        3 * generators[0] - 2 * generators[1] + 4 * generators[2] - generators[3],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle multiple sequences of length zero") {
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = 0, .data = nullptr},
        {.element_nbytes = 1, .n = 0, .data = nullptr}};
    auto res = f({}, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        c21t::element_p3::identity(),
        c21t::element_p3::identity(),
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle multiple sequences with a single exponent") {
    std::vector<uint8_t> exponents1 = {1};
    std::vector<uint16_t> exponents2 = {3};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents1),
        mtxb::to_exponent_sequence(exponents2),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = f(generators, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        generators[0],
        3 * generators[0],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle multiple sequences with multiple exponents") {
    std::vector<uint8_t> exponents1 = {1, 10, 3};
    std::vector<uint16_t> exponents2 = {2, 6, 4};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents1),
        mtxb::to_exponent_sequence(exponents2),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
        0x456_rs,
        0x789_rs,
    };
    auto res = f(generators, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        generators[0] + 10 * generators[1] + 3 * generators[2],
        2 * generators[0] + 6 * generators[1] + 4 * generators[2],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle multiple sequences with multiple exponents where one is all zero") {
    std::vector<uint8_t> exponents1 = {0, 0, 0};
    std::vector<uint16_t> exponents2 = {2, 6, 4};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents1),
        mtxb::to_exponent_sequence(exponents2),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
        0x456_rs,
        0x789_rs,
    };
    auto res = f(generators, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        c21t::element_p3::identity(),
        2 * generators[0] + 6 * generators[1] + 4 * generators[2],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle multiple sequences  of varying length") {
    std::vector<uint8_t> exponents1 = {10};
    std::vector<uint16_t> exponents2 = {2, 6, 4};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::to_exponent_sequence(exponents1),
        mtxb::to_exponent_sequence(exponents2),
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
        0x456_rs,
        0x789_rs,
    };
    auto res = f(generators, sequences);
    memmg::managed_array<c21t::element_p3> expected = {
        10 * generators[0],
        2 * generators[0] + 6 * generators[1] + 4 * generators[2],
    };
    REQUIRE(res == expected);
  }

  // random test cases
  std::pmr::monotonic_buffer_resource resource;
  basct::span<mtxb::exponent_sequence> sequences;
  basct::span<c21t::element_p3> generators;

  SECTION("we handle random sequences of varying length") {
    mtxrn::random_multiexponentiation_descriptor descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 1,
        .min_sequence_length = 0,
        .max_sequence_length = 100,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 1,
    };
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = f(generators, sequences);
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION("we handle multiple products and random sequences of length 1") {
    mtxrn::random_multiexponentiation_descriptor descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .min_sequence_length = 1,
        .max_sequence_length = 1,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 1,
    };
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = f(generators, sequences);
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION("we handle multiple products and random sequences of varying length") {
    mtxrn::random_multiexponentiation_descriptor descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .min_sequence_length = 1,
        .max_sequence_length = 100,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 1,
    };
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = f(generators, sequences);
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION("we handle random sequences of length 1 and varying num_bytes") {
    mtxrn::random_multiexponentiation_descriptor descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 1,
        .min_sequence_length = 1,
        .max_sequence_length = 1,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 32,
    };
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = f(generators, sequences);
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION("we handle random sequences of varying length and varying num_bytes") {
    mtxrn::random_multiexponentiation_descriptor descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 1,
        .min_sequence_length = 1,
        .max_sequence_length = 100,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 32,
    };
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = f(generators, sequences);
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION("we handle multiple products of random sequences of varying length and varying "
          "num_bytes") {
    mtxrn::random_multiexponentiation_descriptor descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .min_sequence_length = 1,
        .max_sequence_length = 100,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 32,
    };
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = f(generators, sequences);
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION("we handle multiple products of random sequences of varying signs") {
    mtxrn::random_multiexponentiation_descriptor descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .min_sequence_length = 1,
        .max_sequence_length = 100,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 1,
    };
    size_t index = 0;
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      for (auto& seq : sequences) {
        seq.is_signed = index++ % 2;
      }
      auto res = f(generators, sequences);
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION(
      "we handle multiple products of random sequences of varying signs and varying num_bytes") {
    mtxrn::random_multiexponentiation_descriptor descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .min_sequence_length = 1,
        .max_sequence_length = 100,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 1,
    };
    size_t index = 0;
    for (int i = 0; i < 10; ++i) {
      auto num_bytes = 1ull << (i % 5);
      descriptor.min_exponent_num_bytes = num_bytes;
      descriptor.max_exponent_num_bytes = num_bytes;
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      for (auto& seq : sequences) {
        seq.is_signed = index++ % 2;
      }
      auto res = f(generators, sequences);
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }
}
} // namespace sxt::mtxtst
