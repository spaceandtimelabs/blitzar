#include "sxt/scalar25/type/literal.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::s25t;

TEST_CASE("literals are valid") {
  REQUIRE(element{0u} == 0x00_s25);
  REQUIRE(element{3u} == 0x3_s25);

  // 2^252 + 27742317777372353535851937790883648493
  element e;
  std::array<uint64_t, 4> s = {6346243789798364141ull, 1503914060200516822ull, 0ull,
                               1152921504606846976ull};
  memcpy(e.data(), s.data(), 32);
  REQUIRE(e == 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed_s25);
}
