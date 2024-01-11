#include "sxt/multiexp/bucket_method/count.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/chained_resource.h"
#include "sxt/memory/resource/pinned_resource.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// count_bucket_entries_kernel
//--------------------------------------------------------------------------------------------------
static __global__ void count_bucket_entries_kernel(unsigned* count_array,
                                                   const uint8_t* const* scalars,
                                                   unsigned element_num_bytes, unsigned bit_width,
                                                   unsigned n) {
  unsigned output_index = threadIdx.x;
  unsigned bucket_group_index = threadIdx.y;
  unsigned partition_index = blockIdx.x;
  unsigned num_partitions = gridDim.x;
  unsigned num_buckets_per_group = 1u << bit_width;
  unsigned num_bucket_groups = blockDim.y;
  auto output_scalars = scalars[output_index];
  auto bucket_counts = count_array;
  bucket_counts += output_index * num_buckets_per_group * num_buckets_per_group * num_partitions;
  bucket_counts += bucket_group_index * num_buckets_per_group * num_partitions;
  bucket_counts += partition_index * num_buckets_per_group;
  for (unsigned count_index = 0; count_index < num_buckets_per_group; ++count_index) {
    bucket_counts[count_index] = 0;
  }
  (void)output_scalars;
  (void)output_index;
  (void)bucket_group_index;
  (void)partition_index;
  (void)count_array;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
  (void)n;
}

//--------------------------------------------------------------------------------------------------
// count_bucket_entries
//--------------------------------------------------------------------------------------------------
void count_bucket_entries(memmg::managed_array<unsigned>& count_array, const basdv::stream& stream,
                          basct::cspan<uint8_t*> scalars, unsigned n, unsigned element_num_bytes,
                          unsigned bit_width, unsigned num_partitions) noexcept {
  SXT_DEBUG_ASSERT(
      basdv::is_active_device_pointer(scalars.data())
  );
  auto num_bucket_groups = basn::divide_up(element_num_bytes * 8u, bit_width);
  auto num_outputs = scalars.size();
  auto num_buckets_per_group = 1u << bit_width;
  count_array.resize(num_outputs * num_bucket_groups * num_buckets_per_group * num_partitions);
  SXT_DEBUG_ASSERT(basdv::is_active_device_pointer(count_array.data()));
  memr::async_device_resource resource{stream};
  count_bucket_entries_kernel<<<num_partitions, dim3(num_outputs, num_bucket_groups, 1), 0,
                                stream>>>(count_array.data(), scalars.data(), element_num_bytes,
                                          bit_width, n);
}
} // namespace sxt::mtxbk
