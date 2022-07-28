#include "sxt/seqcommit/generator/gpu_generator.h"

#include "sxt/base/container/span.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/seqcommit/generator/base_element.h"

namespace sxt::sqcgn {

static constexpr uint64_t block_size = 64;

//--------------------------------------------------------------------------------------------------
// compute_generators_kernel
//--------------------------------------------------------------------------------------------------
__global__ static void compute_generators_kernel(c21t::element_p3* generators,
                                                 uint64_t num_generators,
                                                 uint64_t offset_generators) {
  int row_i = threadIdx.x + blockIdx.x * blockDim.x;

  if (row_i < num_generators) {
    sqcgn::compute_base_element(generators[row_i], row_i + offset_generators);
  }
}

//--------------------------------------------------------------------------------------------------
// gpu_get_generators
//--------------------------------------------------------------------------------------------------
void gpu_get_generators(basct::span<c21t::element_p3> generators,
                        uint64_t offset_generators) noexcept {

  uint64_t num_generators = generators.size();
  uint64_t num_blocks = basn::divide_up(num_generators, block_size);

  memmg::managed_array<c21t::element_p3> generators_device(num_generators,
                                                           memr::get_managed_device_resource());

  compute_generators_kernel<<<num_blocks, block_size>>>(generators_device.data(), num_generators,
                                                        offset_generators);

  basdv::memcpy_device_to_host(generators.data(), generators_device.data(),
                               generators_device.num_bytes());
}

} // namespace sxt::sqcgn
