#include "sxt/multiexp/curve21/multiexponentiation_cpu_driver.h"

#include <cstdint>
#include <limits>
#include <memory_resource>
#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/overload.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/curve21/naive_multiproduct_solver.h"
#include "sxt/multiexp/pippenger/multiexponentiation2.h"
#include "sxt/multiexp/random/random_multiexponentiation_descriptor.h"
#include "sxt/multiexp/random/random_multiexponentiation_generation.h"
#include "sxt/multiexp/test/curve21_arithmetic.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/literal.h"

using namespace sxt;
using namespace sxt::mtxc21;
using sxt::rstt::operator""_rs;

TEST_CASE("we can compute multiexponentiations with cuve21 elements") {
  naive_multiproduct_solver solver;
  multiexponentiation_cpu_driver drv{&solver};

  SECTION("we handle the empty case") {
    std::vector<mtxb::exponent_sequence> sequences;
    auto res = mtxpi::compute_multiexponentiation(drv, {}, sequences).await_result();
    REQUIRE(res.empty());
  }

  SECTION("we handle a sequence with no exponents") {
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 0, .n = 0, .data = nullptr}};
    auto res = mtxpi::compute_multiexponentiation(drv, {}, sequences)
                   .await_result()
                   .as_array<c21t::element_p3>();
    memmg::managed_array<c21t::element_p3> expected = {
        c21cn::zero_p3_v,
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle an exponent of zero") {
    std::vector<uint32_t> exponents = {0};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                .n = 1,
                                .data = reinterpret_cast<uint8_t*>(exponents.data())}};
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = mtxpi::compute_multiexponentiation(drv, generators, sequences)
                   .await_result()
                   .as_array<c21t::element_p3>();
    memmg::managed_array<c21t::element_p3> expected = {
        c21cn::zero_p3_v,
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle an exponent of one") {
    std::vector<uint32_t> exponents = {1};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                .n = 1,
                                .data = reinterpret_cast<uint8_t*>(exponents.data())}};
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = mtxpi::compute_multiexponentiation(drv, generators, sequences)
                   .await_result()
                   .as_array<c21t::element_p3>();
    REQUIRE(res == generators);
  }

  SECTION("we handle an exponent of two") {
    std::vector<uint32_t> exponents = {2};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                .n = 1,
                                .data = reinterpret_cast<uint8_t*>(exponents.data())}};
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = mtxpi::compute_multiexponentiation(drv, generators, sequences)
                   .await_result()
                   .as_array<c21t::element_p3>();
    memmg::managed_array<c21t::element_p3> expected = {
        generators[0] + generators[0],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle an exponent of three") {
    std::vector<uint32_t> exponents = {3};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                .n = 1,
                                .data = reinterpret_cast<uint8_t*>(exponents.data())}};
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = mtxpi::compute_multiexponentiation(drv, generators, sequences)
                   .await_result()
                   .as_array<c21t::element_p3>();
    memmg::managed_array<c21t::element_p3> expected = {
        generators[0] + generators[0] + generators[0],
    };
    REQUIRE(res == expected);
  }

  SECTION("we can handle a large exponent") {
    std::vector<uint64_t> exponents = {std::numeric_limits<uint64_t>::max()};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                .n = 1,
                                .data = reinterpret_cast<uint8_t*>(exponents.data())}};
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = mtxpi::compute_multiexponentiation(drv, generators, sequences)
                   .await_result()
                   .as_array<c21t::element_p3>();
    memmg::managed_array<c21t::element_p3> expected = {
        exponents[0] * generators[0],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle sequences with more than a single element") {
    std::vector<uint16_t> exponents = {3, 2};
    std::vector<mtxb::exponent_sequence> sequences = {
        mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                .n = exponents.size(),
                                .data = reinterpret_cast<uint8_t*>(exponents.data())}};
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
        0x456_rs,
    };
    auto res = mtxpi::compute_multiexponentiation(drv, generators, sequences)
                   .await_result()
                   .as_array<c21t::element_p3>();
    memmg::managed_array<c21t::element_p3> expected = {
        3 * generators[0] + 2 * generators[1],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle multiple sequences of length zero") {
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 0, .n = 0, .data = nullptr},
        {.element_nbytes = 0, .n = 0, .data = nullptr}};
    auto res = mtxpi::compute_multiexponentiation(drv, {}, sequences)
                   .await_result()
                   .as_array<c21t::element_p3>();
    memmg::managed_array<c21t::element_p3> expected = {
        c21cn::zero_p3_v,
        c21cn::zero_p3_v,
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle multiple sequences with a single exponent") {
    std::vector<uint8_t> exponents1 = {1};
    std::vector<uint16_t> exponents2 = {3};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = sizeof(exponents1[0]),
         .n = exponents1.size(),
         .data = reinterpret_cast<const uint8_t*>(exponents1.data())},
        {.element_nbytes = sizeof(exponents2[0]),
         .n = exponents2.size(),
         .data = reinterpret_cast<const uint8_t*>(exponents2.data())},
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
    };
    auto res = mtxpi::compute_multiexponentiation(drv, generators, sequences)
                   .await_result()
                   .as_array<c21t::element_p3>();
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
        {.element_nbytes = sizeof(exponents1[0]),
         .n = exponents1.size(),
         .data = reinterpret_cast<const uint8_t*>(exponents1.data())},
        {.element_nbytes = sizeof(exponents2[0]),
         .n = exponents2.size(),
         .data = reinterpret_cast<const uint8_t*>(exponents2.data())},
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
        0x456_rs,
        0x789_rs,
    };
    auto res = mtxpi::compute_multiexponentiation(drv, generators, sequences)
                   .await_result()
                   .as_array<c21t::element_p3>();
    memmg::managed_array<c21t::element_p3> expected = {
        generators[0] + 10 * generators[1] + 3 * generators[2],
        2 * generators[0] + 6 * generators[1] + 4 * generators[2],
    };
    REQUIRE(res == expected);
  }

  SECTION("we handle multiple sequences  of varying length") {
    std::vector<uint8_t> exponents1 = {10};
    std::vector<uint16_t> exponents2 = {2, 6, 4};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = sizeof(exponents1[0]),
         .n = exponents1.size(),
         .data = reinterpret_cast<const uint8_t*>(exponents1.data())},
        {.element_nbytes = sizeof(exponents2[0]),
         .n = exponents2.size(),
         .data = reinterpret_cast<const uint8_t*>(exponents2.data())},
    };
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
        0x456_rs,
        0x789_rs,
    };
    auto res = mtxpi::compute_multiexponentiation(drv, generators, sequences)
                   .await_result()
                   .as_array<c21t::element_p3>();
    memmg::managed_array<c21t::element_p3> expected = {
        10 * generators[0],
        2 * generators[0] + 6 * generators[1] + 4 * generators[2],
    };
    REQUIRE(res == expected);
  }
}

TEST_CASE("we can compute randomized multiexponentiations with cuve21 elements") {
  naive_multiproduct_solver solver;
  multiexponentiation_cpu_driver drv{&solver};
  std::mt19937 rng{2022};

  std::pmr::monotonic_buffer_resource resource;
  basct::span<mtxb::exponent_sequence> sequences;
  basct::span<c21t::element_p3> generators;

  SECTION("we handle multiple random sequences of varying length") {
    mtxrn::random_multiexponentiation_descriptor2 descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 1,
        .min_sequence_length = 0,
        .max_sequence_length = 100,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 1,
    };
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences)
                     .await_result()
                     .as_array<c21t::element_p3>();
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION("we handle multiple outputs and random sequences of length 1") {
    mtxrn::random_multiexponentiation_descriptor2 descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .min_sequence_length = 1,
        .max_sequence_length = 1,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 1,
    };
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences)
                     .await_result()
                     .as_array<c21t::element_p3>();
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION("we handle multiple outputs and multiple random sequences of varying length") {
    mtxrn::random_multiexponentiation_descriptor2 descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .min_sequence_length = 1,
        .max_sequence_length = 100,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 1,
    };
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences)
                     .await_result()
                     .as_array<c21t::element_p3>();
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION("we handle random sequences of length 1 and varying num_bytes") {
    mtxrn::random_multiexponentiation_descriptor2 descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 1,
        .min_sequence_length = 1,
        .max_sequence_length = 1,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 32,
    };
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences)
                     .await_result()
                     .as_array<c21t::element_p3>();
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION("we handle multiple random sequences of varying length and varying num_bytes") {
    mtxrn::random_multiexponentiation_descriptor2 descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 1,
        .min_sequence_length = 1,
        .max_sequence_length = 100,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 32,
    };
    for (int i = 0; i < 10; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences)
                     .await_result()
                     .as_array<c21t::element_p3>();
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }

  SECTION("we handle multiple outputs and multiple random sequences of varying length and varying "
          "num_bytes") {
    mtxrn::random_multiexponentiation_descriptor2 descriptor{
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .min_sequence_length = 1,
        .max_sequence_length = 100,
        .min_exponent_num_bytes = 1,
        .max_exponent_num_bytes = 32,
    };
    for (int i = 0; i < 100; ++i) {
      mtxrn::generate_random_multiexponentiation(generators, sequences, &resource, rng, descriptor);
      auto res = compute_multiexponentiation(drv, generators, sequences)
                     .await_result()
                     .as_array<c21t::element_p3>();
      memmg::managed_array<c21t::element_p3> expected(sequences.size());
      mtxtst::mul_sum_curve21_elements(expected, generators, sequences);
      REQUIRE(res == expected);
    }
  }
}
