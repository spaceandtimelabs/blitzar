#include "sxt/base/error/assert.h"

#include "sxt/base/test/unit_test.h"

TEST_CASE("We can use release assert") {
  SXT_RELEASE_ASSERT(1);
  SXT_RELEASE_ASSERT(1, "Should not fail");
  int a = 3, b = 5;
  SXT_RELEASE_ASSERT(a + 2 == b);
  SXT_RELEASE_ASSERT(a + 2 == b, "Should not fail");
}

TEST_CASE("We can use debug assert") {
  SXT_DEBUG_ASSERT(1);
  SXT_DEBUG_ASSERT(1, "Should not fail");
  int a = 3, b = 5;
  SXT_DEBUG_ASSERT(a + 2 == b);
  SXT_DEBUG_ASSERT(a + 2 == b, "Should not fail");
}
