#include "sxt/multiexp/pippenger/multiexponentiation.h"

#include <vector>

#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/pippenger/test_driver.h"
using namespace sxt;
using namespace sxt::mtxpi;

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

  SECTION("we handle multiple exponents") {
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

  SECTION("we handle multiple outputs and multiple exponents") {
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

  SECTION("we handle exponents of varying length") {
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
}
