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
  init_precomputed_generators(10, false);
  generators = get_precomputed_generators();
  REQUIRE(generators.size() == 10);

  // the precomputed generators match the computed values
  c21t::element_p3 e;
  compute_base_element(e, 0);
  REQUIRE(generators[0] == e);

  compute_base_element(e, 9);
  REQUIRE(generators[9] == e);

  std::vector<c21t::element_p3> data;
  generators = get_precomputed_generators(data, 10, 0, false);
  REQUIRE(data.empty());
  compute_base_element(e, 9);
  REQUIRE(generators[9] == e);

  // we get correct generators when `data.length() > precomputed.length()`
  generators = get_precomputed_generators(data, 12, 0, false);
  REQUIRE(data.size() == 12);
  compute_base_element(e, 11);
  REQUIRE(generators[11] == e);

  // we get correct generators when `offset != 0` and
  // `offset + data.length() <= precomputed.length()`
  generators = get_precomputed_generators(data, 2, 4, false);
  REQUIRE(data.size() == 12); // data should not be modified
  compute_base_element(e, 4);
  REQUIRE(generators[0] == e);

  generators = get_precomputed_generators(data, 6, 4, false);
  REQUIRE(data.size() == 12); // data should not be modified
  compute_base_element(e, 4);
  REQUIRE(generators[0] == e);
  compute_base_element(e, 9);
  REQUIRE(generators[5] == e);

  // we get correct generators when `offset != 0` and `offset < precomputed.length()`,
  // but `offset + data.length() > precomputed.length()`
  generators = get_precomputed_generators(data, 8, 3, false);
  REQUIRE(data.size() == 8);
  compute_base_element(e, 3);
  REQUIRE(generators[0] == e);
  compute_base_element(e, 10);
  REQUIRE(generators[7] == e);

  // we get correct generators when `offset > precomputed.length()`
  generators = get_precomputed_generators(data, 2, 12, false);
  REQUIRE(data.size() == 2);
  compute_base_element(e, 12);
  REQUIRE(generators[0] == e);
  compute_base_element(e, 13);
  REQUIRE(generators[1] == e);
}
