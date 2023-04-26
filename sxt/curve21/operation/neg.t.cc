#include "sxt/curve21/operation/neg.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/literal.h"

using namespace sxt;
using namespace sxt::c21o;
using c21t::operator""_c21;

TEST_CASE("we can negate curve-21 elements") {
  auto p = 0x123_c21;
  c21t::element_p3 np;
  neg(np, p);
  c21t::element_p3 z;
  add(z, p, np);
  auto q = 0x456_c21;
  c21t::element_p3 qp;
  add(qp, q, z);
  REQUIRE(q == qp);
}
