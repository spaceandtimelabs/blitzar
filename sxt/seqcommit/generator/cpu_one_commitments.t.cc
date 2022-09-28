#include "sxt/seqcommit/generator/cpu_one_commitments.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/cpu_generator.h"

using namespace sxt;
using namespace sxt::sqcgn;

TEST_CASE("test one commitments") {
  std::vector<c21t::element_p3> precomputed_values(5), generators(5);

  cpu_get_generators(generators, 0);
  cpu_get_one_commitments(precomputed_values);

  c21t::element_p3 sum_gen_0_1;
  c21o::add(sum_gen_0_1, generators[0], generators[1]);

  SECTION("we can correctly generate precomputed values") {
    REQUIRE(c21cn::zero_p3_v == precomputed_values[0]);
    REQUIRE(generators[0] == precomputed_values[1]);
    REQUIRE(sum_gen_0_1 == precomputed_values[2]);
  }

  // we can compute `cpu_get_one_commit`
  SECTION("we can correctly generate one commitments at i-th position out of identity value") {
    REQUIRE(c21cn::zero_p3_v == cpu_get_one_commit(0));
    REQUIRE(generators[0] == cpu_get_one_commit(1));
    REQUIRE(sum_gen_0_1 == cpu_get_one_commit(2));
  }

  // we can compute `cpu_get_one_commit` with predefined commitments and offsets
  SECTION("we can correctly generate one commitments at i-th position out of a predefined "
          "commitment and offset") {
    REQUIRE(c21cn::zero_p3_v == cpu_get_one_commit(c21cn::zero_p3_v, 0, 0));
    REQUIRE(c21cn::zero_p3_v == cpu_get_one_commit(c21cn::zero_p3_v, 0, 1));
    REQUIRE(generators[0] == cpu_get_one_commit(c21cn::zero_p3_v, 1, 0));
    REQUIRE(generators[0] == cpu_get_one_commit(generators[0], 0, 0));
    REQUIRE(generators[0] == cpu_get_one_commit(generators[0], 0, 1));
    REQUIRE(sum_gen_0_1 == cpu_get_one_commit(c21cn::zero_p3_v, 2, 0));
    REQUIRE(sum_gen_0_1 == cpu_get_one_commit(generators[0], 1, 1));
  }
}
