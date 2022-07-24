#include "sxt/seqcommit/generator/cpu_generator.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/seqcommit/test/test_generators.h"

using namespace sxt;
using namespace sxt::sqcgn;

TEST_CASE("run computation tests") {
  sqctst::test_pedersen_get_generators(
      [](basct::span<rstt::compressed_element> generators, uint64_t offset) noexcept {
        cpu_get_generators(generators, offset);
      });
}
