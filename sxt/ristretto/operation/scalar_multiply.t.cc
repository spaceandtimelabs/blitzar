#include "sxt/ristretto/operation/scalar_multiply.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/ristretto/operation/add.h"
#include "sxt/ristretto/type/compressed_element.h"

using namespace sxt;
using namespace sxt::rsto;

TEST_CASE("we can multiply elements by a scalar") {
  rstt::compressed_element g{226, 242, 174, 10,  106, 188, 78,  113, 168, 132, 169,
                             97,  197, 0,   81,  95,  88,  227, 11,  106, 165, 130,
                             221, 141, 182, 166, 89,  69,  224, 141, 45,  118};

  rstt::compressed_element res;

  SECTION("verify multiply by 1") {
    scalar_multiply(res, 1u, g);
    REQUIRE(res == g);
  }

  SECTION("verify multiply by 2") {
    unsigned char a[32] = {2};
    scalar_multiply(res, a, g);

    rstt::compressed_element expected_res{106, 73,  50,  16,  247, 73,  156, 209, 127, 236, 181,
                                          16,  174, 12,  234, 35,  161, 16,  232, 213, 185, 1,
                                          248, 172, 173, 211, 9,   92,  115, 163, 185, 25};

    REQUIRE(res == expected_res);
  }

  SECTION("verify we can multiply by an exponent with a[31] > 127") {
    uint8_t a1[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125,
    };
    uint8_t a2[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125,
    };
    uint8_t a3[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 250,
    };
    REQUIRE(a3[31] > 127);
    scalar_multiply(res, a1, g);
    auto expected_res = res;

    scalar_multiply(res, a2, g);
    rsto::add(expected_res, expected_res, res);

    scalar_multiply(res, a3, g);
    REQUIRE(res == expected_res);
  }
}
