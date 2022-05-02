#include "sxt/curve21/ristretto/byte_conversion.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"

using namespace sxt;
using namespace sxt::c21rs;

TEST_CASE("Testing point conversion from curve255 to ristretto") {
  c21t::element_p3 p;

  SECTION("verify against values from libsodium") {
    uint8_t s[32];

    to_bytes(s, p);

    // REQUIRE(c21p::is_on_curve(p)); --> Is that required?

    uint8_t expected_s[32] = {
        0, 0, 0
    };

    REQUIRE(s == expected_s);
  }
}
