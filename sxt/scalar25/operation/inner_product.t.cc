#include "sxt/scalar25/operation/inner_product.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::s25o;
using sxt::s25t::operator""_s25;

TEST_CASE("we can compute the inner product of two scalar vectors") {
  s25t::element res;

  SECTION("we properly handle an inner product of vectors consisting of only a single element") {
    std::vector<s25t::element> lhs = {0x3_s25};
    std::vector<s25t::element> rhs = {0x2_s25};
    inner_product(res, lhs, rhs);
    REQUIRE(res == 0x6_s25);
  }

  SECTION("we handle vectors of more than a single element") {
    std::vector<s25t::element> lhs = {0x3_s25, 0x123_s25, 0x456_s25};
    std::vector<s25t::element> rhs = {0x2_s25, 0x9234_s25, 0x6435_s25};
    inner_product(res, lhs, rhs);
    auto expected = lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
    REQUIRE(res == expected);
  }

  SECTION("if vectors are of unequal length, we compute the product as if the smaller vector was "
          "padded with zeros") {
    std::vector<s25t::element> lhs = {0x3_s25, 0x123_s25, 0x456_s25};
    std::vector<s25t::element> rhs = {0x2_s25};
    inner_product(res, lhs, rhs);
    auto expected = lhs[0] * rhs[0];
    REQUIRE(res == expected);
    inner_product(res, rhs, lhs);
    REQUIRE(res == expected);
  }
}
