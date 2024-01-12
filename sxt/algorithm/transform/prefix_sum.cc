#include "sxt/algorithm/transform/prefix_sum.h"

#include <exception> // https://github.com/NVIDIA/cccl/issues/1278
#include <cstddef>

#include "cub/cub.cuh"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::algtr {
//--------------------------------------------------------------------------------------------------
// exclusive_prefix_sum 
//--------------------------------------------------------------------------------------------------
void exclusive_prefix_sum(basct::span<unsigned> out, basct::span<unsigned> in,
                           const basdv::stream& stream) noexcept {
  auto n = static_cast<int>(out.size());
  if (n == 0) {
    return;
  }
  SXT_DEBUG_ASSERT(
      // clang-format off
      out.size() == static_cast<size_t>(n) &&
      in.size() == static_cast<size_t>(n) &&
      basdv::is_active_device_pointer(out.data()) &&
      basdv::is_active_device_pointer(in.data())
      // clang-format on
  )

  memr::async_device_resource resource{stream};

  // query amount of memory needed
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, in.data(), out.data(), n, stream);

  // compute prefix sum
  memmg::managed_array<std::byte> temp_storage{temp_storage_bytes, &resource};
  cub::DeviceScan::ExclusiveSum(temp_storage.data(), temp_storage_bytes, in.data(), out.data(), n,
                                stream);
}
} // namespace sxt::algtr
