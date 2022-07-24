#include "sxt/seqcommit/generator/precomputed_generators.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/base_element.h"

using namespace sxt;
using namespace sxt::sqcgn;

TEST_CASE("we can precompute generators") {
  // we start with no generators precomputed
  auto generators = get_precomputed_generators();
  REQUIRE(generators.empty());

  // if we precompute generators, we can access them
  init_precomputed_generators(10);
  generators = get_precomputed_generators();
  REQUIRE(generators.size() == 10);

  // the precomputed generators match the computed values
  c21t::element_p3 e;
  compute_base_element(e, 0);
  REQUIRE(generators[0] == e);

  compute_base_element(e, 9);
  REQUIRE(generators[9] == e);
}
