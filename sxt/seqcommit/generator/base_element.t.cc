#include "sxt/seqcommit/generator/base_element.h"

#include "sxt/base/test/unit_test.h"

#include "sxt/curve21/type/element_p3.h"

using namespace sxt;
using namespace sxt::sqcgn;

TEST_CASE("we can deterministically generate base elements for a given row index") {
  c21t::element_p3 p1, p2;

  SECTION("we generate different base elements for different indexes") {
    compute_base_element(p1, 0);
    compute_base_element(p2, 1);
    REQUIRE(p1 != p2);
  }

  SECTION("we generate the same base element for the same row index") {
    compute_base_element(p1, 1);
    compute_base_element(p2, 1);
    REQUIRE(p1 == p2);
  }
}
