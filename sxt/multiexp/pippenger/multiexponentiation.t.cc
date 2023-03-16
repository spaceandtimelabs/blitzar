#include "sxt/multiexp/pippenger/multiexponentiation.h"

#include <iostream>
#include <memory_resource>
#include <random>
#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/pippenger/test_driver.h"
#include "sxt/multiexp/random/random_multiexponentiation_descriptor.h"
#include "sxt/multiexp/random/random_multiexponentiation_generation.h"
#include "sxt/multiexp/test/compute_uint64_muladd.h"

using namespace sxt;
using namespace sxt::mtxpi;

TEST_CASE("we can compute select multiexponentiations") {
  test_driver drv;

  SECTION("we handle the empty case") {
    std::vector<mtxb::exponent_sequence> sequences;
    auto res = compute_multiexponentiation(drv, {}, sequences);
    REQUIRE(res.value().empty());
  }

  SECTION("we handle the zero multiplier case") {
    memmg::managed_array<uint64_t> generators = {123};
    std::vector<uint8_t> exponents = {0};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    auto res = compute_multiexponentiation(drv, generators, sequences);
    memmg::managed_array<uint64_t> expected = {0};
    REQUIRE(res.value().as_array<uint64_t>() == expected);
  }

  SECTION("we handle the 1 multiplier case") {
    memmg::managed_array<uint64_t> generators = {123};
    std::vector<uint8_t> exponents = {1};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    auto res = compute_multiexponentiation(drv, generators, sequences);
    memmg::managed_array<uint64_t> expected = {123};
    REQUIRE(res.value().as_array<uint64_t>() == expected);
  }

  SECTION("we handle the 2 multiplier case") {
    memmg::managed_array<uint64_t> generators = {123};
    std::vector<uint8_t> exponents = {2};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    auto res = compute_multiexponentiation(drv, generators, sequences);
    memmg::managed_array<uint64_t> expected = {2 * 123};
    REQUIRE(res.value().as_array<uint64_t>() == expected);
  }

  SECTION("we handle zero exponents") {
    memmg::managed_array<uint64_t> generators = {123, 456, 789};
    std::vector<uint8_t> exponents = {2, 0, 1};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = exponents.size(), .data = exponents.data()}};
    auto res = compute_multiexponentiation(drv, generators, sequences);
    memmg::managed_array<uint64_t> expected = {2 * 123 + 789};
    REQUIRE(res.value().as_array<uint64_t>() == expected);
  }

  SECTION("we handle a large multiplier") {
    memmg::managed_array<uint64_t> generators = {123};
    std::vector<uint64_t> exponents = {1'000'000};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = sizeof(uint64_t),
         .n = 1,
         .data = reinterpret_cast<const uint8_t*>(exponents.data())}};
    auto res = compute_multiexponentiation(drv, generators, sequences);
    memmg::managed_array<uint64_t> expected = {123'000'000};
    REQUIRE(res.value().as_array<uint64_t>() == expected);
  }

  SECTION("we handle multiple sequences") {
    memmg::managed_array<uint64_t> generators = {123, 321};
    std::vector<uint8_t> exponents = {2, 3};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = exponents.size(), .data = exponents.data()}};
    auto res = compute_multiexponentiation(drv, generators, sequences);
    memmg::managed_array<uint64_t> expected = {2 * 123 + 3 * 321};
    REQUIRE(res.value().as_array<uint64_t>() == expected);
  }

  SECTION("we handle multiple outputs") {
    memmg::managed_array<uint64_t> generators = {123};
    std::vector<uint8_t> exponents1 = {2};
    std::vector<uint8_t> exponents2 = {3};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = exponents1.size(), .data = exponents1.data()},
        {.element_nbytes = 1, .n = exponents2.size(), .data = exponents2.data()},
    };
    auto res = compute_multiexponentiation(drv, generators, sequences);
    memmg::managed_array<uint64_t> expected = {
        2 * 123,
        3 * 123,
    };
    REQUIRE(res.value().as_array<uint64_t>() == expected);
  }

  SECTION("we handle multiple outputs and multiple sequences") {
    memmg::managed_array<uint64_t> generators = {123, 321};
    std::vector<uint8_t> exponents1 = {2, 10};
    std::vector<uint8_t> exponents2 = {3, 20};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = exponents1.size(), .data = exponents1.data()},
        {.element_nbytes = 1, .n = exponents2.size(), .data = exponents2.data()},
    };
    auto res = compute_multiexponentiation(drv, generators, sequences);
    memmg::managed_array<uint64_t> expected = {
        2 * 123 + 10 * 321,
        3 * 123 + 20 * 321,
    };
    REQUIRE(res.value().as_array<uint64_t>() == expected);
  }

  SECTION("we handle sequences of varying length") {
    memmg::managed_array<uint64_t> generators = {123, 321};
    std::vector<uint8_t> exponents1 = {2};
    std::vector<uint8_t> exponents2 = {3, 20};
    std::vector<uint8_t> exponents3 = {10};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = exponents1.size(), .data = exponents1.data()},
        {.element_nbytes = 1, .n = exponents2.size(), .data = exponents2.data()},
        {.element_nbytes = 1, .n = exponents3.size(), .data = exponents3.data()},
    };
    auto res = compute_multiexponentiation(drv, generators, sequences);
    memmg::managed_array<uint64_t> expected = {
        2 * 123,
        3 * 123 + 20 * 321,
        10 * 123,
    };
    REQUIRE(res.value().as_array<uint64_t>() == expected);
  }
}

TEST_CASE("we can compute randomized multiexponentiations") {
  test_driver drv;
  std::mt19937 rng{2022};

  std::pmr::monotonic_buffer_resource resource;
  basct::span<mtxb::exponent_sequence> sequences;
  basct::span<uint64_t> generators;

  SECTION("we handle a random sequences of varying length") {
    mtxrn::random_multiexponentiation_descriptor descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 1,
        .min_sequence_length = 0,
        .max_sequence_length = 100,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 1,
    };
    for (size_t i = 0; i < 1000; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences);
      memmg::managed_array<uint64_t> expected(sequences.size());
      mtxtst::compute_uint64_muladd(expected, generators, sequences);
      REQUIRE(res.value().as_array<uint64_t>() == expected);
    }
  }

  SECTION("we handle multiple outputs and random sequences of length 1") {
    mtxrn::random_multiexponentiation_descriptor descriptor{.min_num_sequences = 1,
                                                            .max_num_sequences = 1'000,
                                                            .min_sequence_length = 1,
                                                            .max_sequence_length = 1,
                                                            .min_exponent_num_bytes = 1,
                                                            .max_exponent_num_bytes = 1};
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences);
      memmg::managed_array<uint64_t> expected(sequences.size());
      mtxtst::compute_uint64_muladd(expected, generators, sequences);
      REQUIRE(res.value().as_array<uint64_t>() == expected);
    }
  }

  SECTION("we handle multiple outputs and with random sequences of varying length") {
    mtxrn::random_multiexponentiation_descriptor descriptor{.min_num_sequences = 1,
                                                            .max_num_sequences = 100,
                                                            .min_sequence_length = 1,
                                                            .max_sequence_length = 100,
                                                            .min_exponent_num_bytes = 1,
                                                            .max_exponent_num_bytes = 1};
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences);
      memmg::managed_array<uint64_t> expected(sequences.size());
      mtxtst::compute_uint64_muladd(expected, generators, sequences);
      REQUIRE(res.value().as_array<uint64_t>() == expected);
    }
  }

  SECTION("we handle random sequences of length 1 and varying num_bytes") {
    mtxrn::random_multiexponentiation_descriptor descriptor{.min_num_sequences = 1,
                                                            .max_num_sequences = 1,
                                                            .min_sequence_length = 1,
                                                            .max_sequence_length = 1,
                                                            .min_exponent_num_bytes = 1,
                                                            .max_exponent_num_bytes = 8};
    for (size_t i = 0; i < 100; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences);
      memmg::managed_array<uint64_t> expected(sequences.size());
      mtxtst::compute_uint64_muladd(expected, generators, sequences);
      REQUIRE(res.value().as_array<uint64_t>() == expected);
    }
  }

  SECTION("we handle random sequences of varying length and varying num_bytes") {
    mtxrn::random_multiexponentiation_descriptor descriptor{.min_num_sequences = 1,
                                                            .max_num_sequences = 1,
                                                            .min_sequence_length = 1,
                                                            .max_sequence_length = 100,
                                                            .min_exponent_num_bytes = 1,
                                                            .max_exponent_num_bytes = 8};
    for (size_t i = 0; i < 100; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences);
      memmg::managed_array<uint64_t> expected(sequences.size());
      mtxtst::compute_uint64_muladd(expected, generators, sequences);
      REQUIRE(res.value().as_array<uint64_t>() == expected);
    }
  }

  SECTION(
      "we handle multiple outputs and random sequences of varying length and varying num_bytes") {
    mtxrn::random_multiexponentiation_descriptor descriptor{.min_num_sequences = 1,
                                                            .max_num_sequences = 100,
                                                            .min_sequence_length = 1,
                                                            .max_sequence_length = 100,
                                                            .min_exponent_num_bytes = 1,
                                                            .max_exponent_num_bytes = 8};
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences);
      memmg::managed_array<uint64_t> expected(sequences.size());
      mtxtst::compute_uint64_muladd(expected, generators, sequences);
      REQUIRE(res.value().as_array<uint64_t>() == expected);
    }
  }
}
