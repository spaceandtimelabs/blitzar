#include "sxt/seqcommit/test/test_generators.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/generator/base_element.h"

namespace sxt::sqctst {
//--------------------------------------------------------------------------------------------------
// test_pedersen_get_generators
//--------------------------------------------------------------------------------------------------
void test_pedersen_get_generators(
    basf::function_ref<void(basct::span<rstt::compressed_element> generators,
        uint64_t offset_generators)> f) {

    SECTION("we can verify that computed generators are correct when offset is zero") {
      c21t::element_p3 expected_g_0, expected_g_1;
      uint64_t num_generators = 2;
      uint64_t offset_generators = 0;
      sqcgn::compute_base_element(expected_g_0, 0 + offset_generators);
      sqcgn::compute_base_element(expected_g_1, 1 + offset_generators);

      rstt::compressed_element generators[num_generators];
      basct::span<rstt::compressed_element> span_generators(generators, num_generators);

      f(span_generators, offset_generators);

      rstt::compressed_element expected_commit_0, expected_commit_1;
      rstb::to_bytes(expected_commit_0.data(), expected_g_0);
      rstb::to_bytes(expected_commit_1.data(), expected_g_1);

      REQUIRE(generators[0] == expected_commit_0);
      REQUIRE(generators[1] == expected_commit_1);
    }

    SECTION("we can verify that computed generators are correct when offset is non zero") {
      c21t::element_p3 expected_g_0, expected_g_1;
      uint64_t num_generators = 2;
      uint64_t offset_generators = 15;
      sqcgn::compute_base_element(expected_g_0, 0 + offset_generators);
      sqcgn::compute_base_element(expected_g_1, 1 + offset_generators);

      rstt::compressed_element generators[num_generators];
      basct::span<rstt::compressed_element> span_generators(generators, num_generators);

      f(span_generators, offset_generators);

      rstt::compressed_element expected_commit_0, expected_commit_1;
      rstb::to_bytes(expected_commit_0.data(), expected_g_0);
      rstb::to_bytes(expected_commit_1.data(), expected_g_1);

      REQUIRE(generators[0] == expected_commit_0);
      REQUIRE(generators[1] == expected_commit_1);
    }
}
}  // namespace sxt::sqctst
