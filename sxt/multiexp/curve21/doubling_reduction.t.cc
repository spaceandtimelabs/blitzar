#include "sxt/multiexp/curve21/doubling_reduction.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/overload.h"
#include "sxt/ristretto/type/literal.h"

using namespace sxt;
using namespace sxt::mtxc21;
using sxt::rstt::operator""_rs;

TEST_CASE("we can perform the doubling reduction step of a multiexponentiation") {
  c21t::element_p3 res;

  SECTION("we handle a single element") {
    uint8_t digit_or_all[] = {1};
    c21t::element_p3 inputs[] = {0x123_rs};
    doubling_reduce(res, digit_or_all, inputs);
    REQUIRE(res == inputs[0]);
  }

  SECTION("we handle a single elements that's doubled") {
    uint8_t digit_or_all[] = {2};
    c21t::element_p3 inputs[] = {0x123_rs};
    doubling_reduce(res, digit_or_all, inputs);
    REQUIRE(res == 2 * inputs[0]);
  }

  SECTION("we handle multiple inputs") {
    uint8_t digit_or_all[] = {3};
    c21t::element_p3 inputs[] = {0x123_rs, 0x345_rs};
    doubling_reduce(res, digit_or_all, inputs);
    REQUIRE(res == 2 * inputs[1] + inputs[0]);
  }
}
