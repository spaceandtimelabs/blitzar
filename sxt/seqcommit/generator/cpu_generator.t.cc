#include "sxt/seqcommit/generator/cpu_generator.h"

#include "sxt/seqcommit/test/test_generators.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::sqcgn;

TEST_CASE("run computation tests") {
  sqctst::test_pedersen_get_generators(cpu_get_generators);
}
