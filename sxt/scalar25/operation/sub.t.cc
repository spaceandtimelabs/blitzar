#include "sxt/scalar25/operation/sub.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt::s25o;
using namespace sxt::s25t;

TEST_CASE("we check that the zero subtracted from zero is zero") {
  element s;
  sub(s, 0x0_s25, 0x0_s25); // s = 0 - 0
  REQUIRE(s == 0x0_s25);

  sub(s, 0x3_s25, 0x3_s25); // s = a - a
  REQUIRE(s == 0x0_s25);
}

TEST_CASE("we check that the commutativity is not valid") {
  element s;
  sub(s, 0x2_s25, 0x0_s25); // s = a - 0
  REQUIRE(s == 0x2_s25);

  sub(s, 0x0_s25, 0x2_s25); // s = 0 - a
  REQUIRE(s == 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3eb_s25);
}

TEST_CASE("we check that very small values have valid subtractions") {
  element s;
  sub(s, 0x7_s25, 0x2_s25);
  REQUIRE(s == 0x5_s25);
}

TEST_CASE(
    "we check that a big value (smaller than L) has a valid subtraction (L = the field order)") {
  element s;
  element a =
      0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed_s25; // a = L / 4 + 3
  sub(s, a, 0x1_s25);
  REQUIRE(s == 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ec_s25);

  sub(s, 0x1_s25, a);
  REQUIRE(s == 0x1_s25);
}

TEST_CASE(
    "we check that a big value (even bigger than L) has a subtraction (L = the field order)") {
  element s;
  element a =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7df_s25; // a = 2 * L + 5
  sub(s, 0x5_s25, a);
  REQUIRE(s == 0x0_s25);
}

TEST_CASE("we correctly subtract A - B when B is the biggest 256bits integer") {
  element s;
  element a = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;

  sub(s, 0x34_s25, a);
  REQUIRE(s == 0x14def9dea2f79cd65812631a5cf5d3f05_s25);
}
