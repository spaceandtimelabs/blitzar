#include "sxt/multiexp/bucket_method/count.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/chained_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/digit_utility.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// count_bucket_entries_kernel
//--------------------------------------------------------------------------------------------------
static __global__ void count_bucket_entries_kernel(unsigned* __restrict__ count_array,
                                                   const uint8_t* __restrict__ scalars,
                                                   unsigned element_num_bytes, unsigned n,
                                                   unsigned bit_width) {
  unsigned output_index = threadIdx.x;
  unsigned bucket_group_index = threadIdx.y;
  unsigned partition_index = blockIdx.x;
  unsigned num_partitions = gridDim.x;
  unsigned num_buckets_per_group = (1u << bit_width) - 1u;
  unsigned num_bucket_groups = blockDim.y;
  auto bucket_counts = count_array;
  bucket_counts += output_index * num_bucket_groups * num_buckets_per_group * num_partitions;
  bucket_counts += bucket_group_index * num_buckets_per_group * num_partitions;
  bucket_counts += partition_index;
  for (unsigned count_index = 0; count_index < num_buckets_per_group; ++count_index) {
    *(bucket_counts + count_index * num_partitions) = 0;
  }
  scalars += output_index * element_num_bytes * n;
  auto byte_width = basn::divide_up(bit_width, 8u);
  for (unsigned i = partition_index; i < n; i += num_partitions) {
    basct::cspan<uint8_t> scalar{
        scalars + i * element_num_bytes,
        element_num_bytes,
    };
    unsigned digit = 0;
    mtxb::extract_digit({reinterpret_cast<uint8_t*>(&digit), byte_width}, scalar, bit_width,
                        bucket_group_index);
    auto count_index = max(digit, 1) - 1;
    *(bucket_counts + count_index * num_partitions) += digit != 0;
  }
}

//--------------------------------------------------------------------------------------------------
// count_bucket_entries
//--------------------------------------------------------------------------------------------------
void count_bucket_entries(memmg::managed_array<unsigned>& count_array, const basdv::stream& stream,
                          basct::cspan<uint8_t> scalars, unsigned element_num_bytes, unsigned n,
                          unsigned num_outputs, unsigned bit_width,
                          unsigned num_partitions) noexcept {
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_active_device_pointer(scalars.data()) &&
      scalars.size() == element_num_bytes * n * num_outputs &&
      0u < bit_width && bit_width <= 16u &&
      num_partitions <= n
      // clang-format on
  );
  auto num_bucket_groups = basn::divide_up(element_num_bytes * 8u, bit_width);
  auto num_buckets_per_group = (1u << bit_width) - 1u;
  count_array.resize(num_outputs * num_bucket_groups * num_buckets_per_group * num_partitions);
  SXT_DEBUG_ASSERT(basdv::is_active_device_pointer(count_array.data()));
  memr::async_device_resource resource{stream};
  count_bucket_entries_kernel<<<num_partitions, dim3(num_outputs, num_bucket_groups, 1), 0,
                                stream>>>(count_array.data(), scalars.data(), element_num_bytes, n,
                                          bit_width);
}
} // namespace sxt::mtxbk