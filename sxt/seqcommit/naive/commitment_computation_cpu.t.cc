#include "sxt/seqcommit/naive/commitment_computation_cpu.h"

#include "sxt/seqcommit/test/test_pedersen.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::sqcnv;

TEST_CASE("run computation tests") {
  sqctst::test_pedersen_compute_commitment(compute_commitments_cpu);
}
