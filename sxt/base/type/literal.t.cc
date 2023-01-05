#include "sxt/base/type/literal.h"

#include <limits>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::bast;

TEST_CASE("we can parse literals into arrays of 64-bit integers") {
  SECTION("we can parse a single literal") {
    std::array<uint64_t, 1> vals = {123};
    parse_literal<1, '0', 'x', '2'>(vals);
    std::array<uint64_t, 1> expected = {2};
    REQUIRE(vals == expected);
  }

  SECTION("we can handle multiple digits") {
    std::array<uint64_t, 1> vals = {123};
    parse_literal<1, '0', 'x', '1', '0'>(vals);
    std::array<uint64_t, 1> expected = {16};
    REQUIRE(vals == expected);
  }

  SECTION("we handle the maximum number") {
    std::array<uint64_t, 1> vals = {123};
    parse_literal<1, '0', 'x', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
                  'F', 'F'>(vals);
    std::array<uint64_t, 1> expected = {std::numeric_limits<uint64_t>::max()};
    REQUIRE(vals == expected);
  }

  SECTION("we handle multiple int64s") {
    std::array<uint64_t, 2> vals = {123};
    parse_literal<2, '0', 'x', '3', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
                  'F', 'F', 'F'>(vals);
    std::array<uint64_t, 2> expected = {std::numeric_limits<uint64_t>::max(), 3};
    REQUIRE(vals == expected);
  }
}
