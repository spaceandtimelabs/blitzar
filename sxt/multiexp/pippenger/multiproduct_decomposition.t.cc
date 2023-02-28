#include "sxt/multiexp/pippenger/multiproduct_decomposition.h"

#include <algorithm>
#include <vector>

#include "sxt/base/container/blob_array.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence_utility.h"

using namespace sxt;
using namespace sxt::mtxpi;

TEST_CASE("we can compute the decomposition that turns a multiexponentiation problem into a "
          "multiproduct problem") {
  memmg::managed_array<unsigned> indexes, product_sizes;
  basct::blob_array output_digit_or_all;

  SECTION("we handle the empty case") {
    compute_multiproduct_decomposition(indexes, product_sizes, output_digit_or_all, {});
    REQUIRE(indexes.empty());
    REQUIRE(product_sizes.empty());
    REQUIRE(output_digit_or_all.empty());
  }

  SECTION("we handle a case with only zeros") {
    std::vector<uint8_t> exponents = {0};
    std::vector<mtxb::exponent_sequence> sequences{
        mtxb::to_exponent_sequence(exponents),
    };
    compute_multiproduct_decomposition(indexes, product_sizes, output_digit_or_all, sequences);
    REQUIRE(indexes.empty());
    REQUIRE(product_sizes.empty());
    REQUIRE(output_digit_or_all.size() == 1);
    REQUIRE(output_digit_or_all[0][0] == 0);
  }

  SECTION("we handle a case with a single bit") {
    std::vector<uint8_t> exponents = {1};
    std::vector<mtxb::exponent_sequence> sequences{
        mtxb::to_exponent_sequence(exponents),
    };
    compute_multiproduct_decomposition(indexes, product_sizes, output_digit_or_all, sequences);
    REQUIRE(indexes == memmg::managed_array<unsigned>{0});
    REQUIRE(product_sizes == memmg::managed_array<unsigned>{1});
    REQUIRE(output_digit_or_all.size() == 1);
    REQUIRE(output_digit_or_all[0][0] == 1);
  }

  SECTION("we handle multiple bits") {
    std::vector<uint8_t> exponents = {0b101};
    std::vector<mtxb::exponent_sequence> sequences{
        mtxb::to_exponent_sequence(exponents),
    };
    compute_multiproduct_decomposition(indexes, product_sizes, output_digit_or_all, sequences);
    REQUIRE(indexes == memmg::managed_array<unsigned>{0, 0});
    REQUIRE(product_sizes == memmg::managed_array<unsigned>{1, 1});
    REQUIRE(output_digit_or_all.size() == 1);
    REQUIRE(output_digit_or_all[0][0] == 0b101);
  }

  SECTION("we handle sequences of length greater than 1") {
    std::vector<uint8_t> exponents = {0b101, 0b100};
    std::vector<mtxb::exponent_sequence> sequences{
        mtxb::to_exponent_sequence(exponents),
    };
    compute_multiproduct_decomposition(indexes, product_sizes, output_digit_or_all, sequences);
    REQUIRE(indexes == memmg::managed_array<unsigned>{0, 0, 1});
    REQUIRE(product_sizes == memmg::managed_array<unsigned>{1, 2});
    REQUIRE(output_digit_or_all.size() == 1);
    REQUIRE(output_digit_or_all[0][0] == 0b101);
  }

  SECTION("we handle multiple sequences") {
    std::vector<uint8_t> exponents1 = {0b101, 0b100};
    std::vector<uint8_t> exponents2 = {0b101, 0b010};
    std::vector<mtxb::exponent_sequence> sequences{
        mtxb::to_exponent_sequence(exponents1),
        mtxb::to_exponent_sequence(exponents2),
    };
    compute_multiproduct_decomposition(indexes, product_sizes, output_digit_or_all, sequences);
    REQUIRE(indexes == memmg::managed_array<unsigned>{0, 0, 1, 0, 1, 0});
    REQUIRE(product_sizes == memmg::managed_array<unsigned>{1, 2, 1, 1, 1});
    REQUIRE(output_digit_or_all.size() == 2);
    REQUIRE(output_digit_or_all[0][0] == 0b101);
    REQUIRE(output_digit_or_all[1][0] == 0b111);
  }

  SECTION("we handle the maximum number of indexes") {
    auto max = std::numeric_limits<uint8_t>::max();
    std::vector<uint8_t> exponents1 = {max, max};
    std::vector<uint8_t> exponents2 = {max, max};
    std::vector<mtxb::exponent_sequence> sequences{
        mtxb::to_exponent_sequence(exponents1),
        mtxb::to_exponent_sequence(exponents2),
    };
    compute_multiproduct_decomposition(indexes, product_sizes, output_digit_or_all, sequences);
    auto expected_indexes = [] {
      memmg::managed_array<unsigned> res(32);
      for (size_t i = 0; i < res.size(); ++i) {
        res[i] = i % 2;
      }
      return res;
    }();
    REQUIRE(indexes == expected_indexes);
    auto expected_preducts = [] {
      memmg::managed_array<unsigned> res(16);
      std::fill(res.begin(), res.end(), 2);
      return res;
    }();
    REQUIRE(output_digit_or_all.size() == 2);
    REQUIRE(output_digit_or_all[0][0] == max);
    REQUIRE(output_digit_or_all[1][0] == max);
  }
}
