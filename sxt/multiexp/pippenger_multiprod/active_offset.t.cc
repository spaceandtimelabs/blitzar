#include "sxt/multiexp/pippenger_multiprod/active_offset.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
using namespace sxt::mtxpmp;

TEST_CASE("we can compute the offset where the active entries start") {
  SECTION("we handle the case of no inactive entries") {
    REQUIRE(compute_active_offset(std::vector<uint64_t>{0, 0}) == 2);
  }

  SECTION("we handle the case of inactive entries") {
    REQUIRE(compute_active_offset(std::vector<uint64_t>{0, 1, 9, 10}) == 3);
  }
}
