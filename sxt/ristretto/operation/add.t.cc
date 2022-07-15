#include "sxt/ristretto/operation/add.h"

#include "sxt/base/test/unit_test.h"

#include "sxt/ristretto/type/compressed_element.h"

using namespace sxt;
using namespace sxt::rsto;

TEST_CASE("we can add curve elements") {
  rstt::compressed_element g{226, 242, 174, 10,  106, 188, 78,  113, 168, 132, 169,
                             97,  197, 0,   81,  95,  88,  227, 11,  106, 165, 130,
                             221, 141, 182, 166, 89,  69,  224, 141, 45,  118};

  rstt::compressed_element res;

  SECTION("adding zero acts as the identity function") {
    add(res, rstt::compressed_element{}, g);
    REQUIRE(res == g);
  }
}
