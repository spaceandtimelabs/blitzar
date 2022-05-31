#include "sxt/seqcommit/generator/gpu_generator.h"

#include <cuda_runtime.h>

#include "sxt/seqcommit/test/test_generators.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::sqcgn;

TEST_CASE("run computation tests") {
  int num_devices;

  auto rcode = cudaGetDeviceCount(&num_devices);

  if (rcode == cudaSuccess) {
    sqctst::test_pedersen_get_generators(gpu_get_generators);
  }
}
