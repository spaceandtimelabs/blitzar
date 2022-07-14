#include "sxt/multiexp/pippenger/multiexponentiation.h"

#include <vector>
#include <random>
#include <iostream>
#include <memory_resource>

#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/pippenger/test_driver.h"
#include "sxt/multiexp/test/compute_uint64_muladd.h"
#include "sxt/multiexp/test/int_generation.h"
#include "sxt/multiexp/random/random_multiexponentiation_descriptor.h"
#include "sxt/multiexp/random/random_multiexponentiation_generation.h"

using namespace sxt;
using namespace sxt::mtxpi;

static void compute_random_test_case(
    std::mt19937 &rng,
    size_t num_sequences,
    mtxrn::random_multiexponentiation_descriptor descriptor,
    const test_driver &drv) {
  
  uint64_t num_inputs = 0;
    
  std::pmr::monotonic_buffer_resource resource;
  std::vector<mtxb::exponent_sequence> sequences(num_sequences);

  mtxrn::generate_random_multiexponentiation(
    num_inputs, sequences, &resource, rng, descriptor
  );

  memmg::managed_array<uint64_t> inout(num_inputs);

  mtxtst::generate_uint64s(inout, rng);
  
  memmg::managed_array<uint64_t> expected_result(num_sequences);

  mtxtst::compute_uint64_muladd(expected_result, inout, sequences);

  compute_multiexponentiation(inout, drv, sequences);
  
  REQUIRE(inout == expected_result);
}

TEST_CASE("we can compute multiexponentiations") {
  test_driver drv;

  SECTION("we handle the empty case") {
    memmg::managed_array<uint64_t> inout;
    std::vector<mtxb::exponent_sequence> sequences;
    compute_multiexponentiation(inout, drv, sequences);
    REQUIRE(inout.empty());
  }

  SECTION("we handle the zero multiplier case") {
    memmg::managed_array<uint64_t> inout = {123};
    std::vector<uint8_t> exponents = {0};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    compute_multiexponentiation(inout, drv, sequences);
    memmg::managed_array<uint64_t> expected_result = {0};
    REQUIRE(inout == expected_result);
  }

  SECTION("we handle the 1 multiplier case") {
    memmg::managed_array<uint64_t> inout = {123};
    std::vector<uint8_t> exponents = {1};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    compute_multiexponentiation(inout, drv, sequences);
    memmg::managed_array<uint64_t> expected_result = {123};
    REQUIRE(inout == expected_result);
  }

  SECTION("we handle the 2 multiplier case") {
    memmg::managed_array<uint64_t> inout = {123};
    std::vector<uint8_t> exponents = {2};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    compute_multiexponentiation(inout, drv, sequences);
    memmg::managed_array<uint64_t> expected_result = {246};
    REQUIRE(inout == expected_result);
  }

  SECTION("we handle a large multiplier") {
    memmg::managed_array<uint64_t> inout = {123};
    std::vector<uint64_t> exponents = {1'000'000};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = sizeof(uint64_t),
         .n = 1,
         .data = reinterpret_cast<const uint8_t*>(exponents.data())}};
    compute_multiexponentiation(inout, drv, sequences);
    memmg::managed_array<uint64_t> expected_result = {123'000'000};
    REQUIRE(inout == expected_result);
  }

  SECTION("we handle multiple sequences") {
    memmg::managed_array<uint64_t> inout = {123, 321};
    std::vector<uint8_t> exponents = {2, 3};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = exponents.size(), .data = exponents.data()}};
    compute_multiexponentiation(inout, drv, sequences);
    memmg::managed_array<uint64_t> expected_result = {1209};
    REQUIRE(inout == expected_result);
  }

  SECTION("we handle multiple outputs") {
    memmg::managed_array<uint64_t> inout = {123};
    std::vector<uint8_t> exponents1 = {2};
    std::vector<uint8_t> exponents2 = {3};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = exponents1.size(), .data = exponents1.data()},
        {.element_nbytes = 1, .n = exponents2.size(), .data = exponents2.data()},
    };
    compute_multiexponentiation(inout, drv, sequences);
    memmg::managed_array<uint64_t> expected_result = {246, 369};
    REQUIRE(inout == expected_result);
  }

  SECTION("we handle multiple outputs and multiple sequences") {
    memmg::managed_array<uint64_t> inout = {123, 321};
    std::vector<uint8_t> exponents1 = {2, 10};
    std::vector<uint8_t> exponents2 = {3, 20};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = exponents1.size(), .data = exponents1.data()},
        {.element_nbytes = 1, .n = exponents2.size(), .data = exponents2.data()},
    };
    compute_multiexponentiation(inout, drv, sequences);
    memmg::managed_array<uint64_t> expected_result = {3456, 6789};
    REQUIRE(inout == expected_result);
  }

  SECTION("we handle sequences of varying length") {
    memmg::managed_array<uint64_t> inout = {123, 321};
    std::vector<uint8_t> exponents1 = {2};
    std::vector<uint8_t> exponents2 = {3, 20};
    std::vector<uint8_t> exponents3 = {10};
    std::vector<mtxb::exponent_sequence> sequences = {
        {.element_nbytes = 1, .n = exponents1.size(), .data = exponents1.data()},
        {.element_nbytes = 1, .n = exponents2.size(), .data = exponents2.data()},
        {.element_nbytes = 1, .n = exponents3.size(), .data = exponents3.data()},
    };
    compute_multiexponentiation(inout, drv, sequences);
    memmg::managed_array<uint64_t> expected_result = {246, 6789, 1230};
    REQUIRE(inout == expected_result);
  }

  std::mt19937 rng{2022};

  SECTION("we handle multiple random sequences of varying length") {
    for (size_t i = 0; i < 1000; ++i) {
      compute_random_test_case(
        rng,
        1,
        {
          .min_sequence_length = 0,
          .max_sequence_length = 100,
          .min_exponent_num_bytes = 1,
          .max_exponent_num_bytes = 1
        },
        drv
      );
    }
  }

  SECTION("we handle multiple outputs and random sequences of length 1") {
    for (size_t i = 0; i < 1000; ++i) {
      compute_random_test_case(
        rng,
        i,
        {
          .min_sequence_length = 1,
          .max_sequence_length = 1,
          .min_exponent_num_bytes = 1,
          .max_exponent_num_bytes = 1
        },
        drv
      );
    }
  }

  SECTION("we handle multiple outputs and"
          "multiple random sequences of varying length") {
    for (size_t i = 0; i < 100; ++i) {
      compute_random_test_case(
        rng,
        i,
        {
          .min_sequence_length = 1,
          .max_sequence_length = 100,
          .min_exponent_num_bytes = 1,
          .max_exponent_num_bytes = 1
        },
        drv
      );
    }
  }

  SECTION("we handle random sequences of length 1 and varying num_bytes") {
    for (size_t i = 0; i < 100; ++i) {
      compute_random_test_case(
        rng,
        1,
        {
          .min_sequence_length = 1,
          .max_sequence_length = 1,
          .min_exponent_num_bytes = 1,
          .max_exponent_num_bytes = 8
        },
        drv
      );
    }
  }

  SECTION("we handle multiple random sequences of"
          "varying length and varying num_bytes") {
    for (size_t i = 0; i < 100; ++i) {
      compute_random_test_case(
        rng,
        1,
        {
          .min_sequence_length = 1,
          .max_sequence_length = 100,
          .min_exponent_num_bytes = 1,
          .max_exponent_num_bytes = 8
        },
        drv
      );
    }
  }

  SECTION("we handle multiple outputs and"
          "multiple random sequences of varying"
          "length and varying num_bytes") {
    for (size_t i = 0; i < 100; ++i) {
      compute_random_test_case(
        rng,
        i,
        {
          .min_sequence_length = 1,
          .max_sequence_length = 100,
          .min_exponent_num_bytes = 1,
          .max_exponent_num_bytes = 8
        },
        drv
      );
    }
  }
}
