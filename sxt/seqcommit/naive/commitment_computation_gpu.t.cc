#include "sxt/seqcommit/naive/commitment_computation_gpu.h"

#include <cuda_runtime.h>

#include "sxt/seqcommit/test/test_commitment_computation.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::sqcnv;

TEST_CASE("run computation tests") {
  int num_devices;

  auto rcode = cudaGetDeviceCount(&num_devices);

  if (rcode == cudaSuccess) {
    sqctst::test_commitment_computation_function(compute_commitments_gpu);
  }
}
