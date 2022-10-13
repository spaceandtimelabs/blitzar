#include "sxt/scalar25/operation/sqmul.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt::s25o;
using namespace sxt::s25t;

TEST_CASE("we check the zero value case") {
  SECTION("with s = 0, n != 0, and a != 0") {
    element s = 0x0_s25;
    sqmul(s, 4, 0x4_s25);
    REQUIRE(s == 0x0_s25);
  }

  SECTION("with s != 0, n != 0, and a = 0") {
    element s = 0x4_s25;
    sqmul(s, 4, 0x0_s25);
    REQUIRE(s == 0x0_s25);
  }

  SECTION("with s != 0, n = 0, and a != 0") {
    element s = 0x4_s25;
    sqmul(s, 0, 0x3_s25);
    REQUIRE(s == 0xC_s25);
  }
}

TEST_CASE("we check the one value case") {
  SECTION("with s = 1, n != 1, and a != 1") {
    element s = 0x1_s25;
    sqmul(s, 4, 0x4_s25);
    REQUIRE(s == 0x4_s25);
  }

  SECTION("with s != 1, n != 1, and a = 1") {
    element s = 0x2_s25;
    sqmul(s, 3, 0x1_s25);
    REQUIRE(s == 0x100_s25);
  }

  SECTION("with s != 1, n = 1, and a != 1") {
    element s = 0x4_s25;
    sqmul(s, 1, 0x3_s25);
    REQUIRE(s == 0x30_s25);
  }
}

TEST_CASE("we check that we can square small values") {
  element s = 0x3_s25;
  sqmul(s, 5, 0x6_s25);
  REQUIRE(s == 0x277fdf4cb57706_s25);
}

TEST_CASE(
    "we check that we can square-multiply a big input, but smaller than L (L = the field order)") {
  SECTION("a = L/4 + 3, but n << L and s << L") {
    element a =
        0x40000000000000000000000000000000537be77a8bde735960498c6973d74fe_s25; // a = L / 4 + 3
    element s = 0x4_s25;
    sqmul(s, 2, a);
    REQUIRE(s == 0x2c0_s25);
  }

  SECTION("s = L/4 + 3, but n << L and a << L") {
    element s =
        0x40000000000000000000000000000000537be77a8bde735960498c6973d74fe_s25; // s = L / 4 + 3
    element a = 0x4_s25;
    sqmul(s, 2, a);
    REQUIRE(s == 0xac000000000000000000000000000000e05cfe1957e5d60032c5a95b6752b48_s25);
  }
}

TEST_CASE("we check that we can square-multiply a big input, bigger than L (L = the field order)") {
  SECTION("a > L but s << L and n << L") {
    element a =
        0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7dd_s25; // a = 2 * L + 3
    element s = 0x2_s25;
    sqmul(s, 2, a);
    REQUIRE(s == element{48ull});
  }

  SECTION("s > L but a << L and n << L") {
    element s =
        0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7dd_s25; // s = 2 * L + 3
    element a = 0x5_s25;
    sqmul(s, 3, a);
    REQUIRE(s == 0x8025_s25);
  }

  SECTION("s > L and a > L and n big") {
    element s =
        0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7dd_s25; // s = 2 * L + 3
    element a =
        0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7df_s25; // a = 2 * L + 5
    sqmul(s, 200, a); // s = (a * (s^(2^200))) % L
    REQUIRE(s == 0xaab54bc1651873d214d6a67540c3d4afa64c047071a321115ec515697a685ba_s25);
  }
}
