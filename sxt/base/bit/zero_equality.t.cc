#include "sxt/base/bit/zero_equality.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basbt;

TEST_CASE("we can determine if a region of memory is zero") {
  unsigned char bytes1[10] = {};
  REQUIRE(is_zero(bytes1, sizeof(bytes1)) == 1);

  unsigned char bytes2[10] = {0, 0, 1};
  REQUIRE(is_zero(bytes2, sizeof(bytes2)) == 0);
}
