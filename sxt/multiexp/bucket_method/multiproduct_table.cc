#include "sxt/multiexp/bucket_method/multiproduct_table.h"

#include "sxt/algorithm/transform/prefix_sum.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/digit_utility.h"
#include "sxt/multiexp/base/scalar_array.h"
#include "sxt/multiexp/bucket_method/bucket_descriptor.h"
#include "sxt/multiexp/bucket_method/count.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// max_num_partitions_v
//--------------------------------------------------------------------------------------------------
const unsigned max_num_partitions_v = 64;

//--------------------------------------------------------------------------------------------------
// fill_index_kernel
//--------------------------------------------------------------------------------------------------
static __global__ void fill_index_kernel(unsigned* __restrict__ indexes,
                                         unsigned* __restrict__ offsets,
                                         const uint8_t* __restrict__ scalars,
                                         unsigned element_num_bytes, unsigned n,
                                         unsigned bit_width) {
  unsigned output_index = threadIdx.x;
  unsigned bucket_group_index = threadIdx.y;
  unsigned partition_index = blockIdx.x;
  unsigned num_partitions = gridDim.x;
  unsigned num_buckets_per_group = (1u << bit_width) - 1;
  unsigned num_bucket_groups = blockDim.y;

  offsets += output_index * num_bucket_groups * num_buckets_per_group * num_partitions;
  offsets += bucket_group_index * num_buckets_per_group * num_partitions;
  offsets += partition_index;

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
    if (digit == 0) {
      continue;
    }
    auto offset_index = digit - 1;
    auto& offset = *(offsets + offset_index * num_partitions);
    indexes[offset++] = i;
  }
}

//--------------------------------------------------------------------------------------------------
// fill_bucket_descriptors_kernel 
//--------------------------------------------------------------------------------------------------
static __global__ void fill_bucket_descriptors_kernel(bucket_descriptor* __restrict__ descriptors,
                                                      const unsigned* __restrict__ offsets,
                                                      unsigned num_partitions,
                                                      unsigned num_buckets) noexcept {
  auto bucket_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (bucket_index < num_buckets) {
    return;
  }
  offsets += bucket_index * num_partitions;
  auto first = *offsets;
  auto num_entries = *(offsets + num_partitions) - first;
  descriptors[bucket_index] = {
      .num_entries = num_entries,
      .bucket_index = bucket_index,
      .entry_first = first,
  };
}

//--------------------------------------------------------------------------------------------------
// fill_multiproduct_indexes 
//--------------------------------------------------------------------------------------------------
xena::future<> fill_multiproduct_indexes(memmg::managed_array<unsigned>& bucket_counts,
                                         memmg::managed_array<unsigned>& indexes,
                                         const basdv::stream& stream,
                                         basct::cspan<const uint8_t*> scalars,
                                         unsigned element_num_bytes, unsigned n,
                                         unsigned bit_width) noexcept {
  if (n == 0) {
    co_return;
  }
  auto num_outputs = scalars.size();
  auto num_bucket_groups = basn::divide_up(element_num_bytes * 8u, bit_width);
  auto num_partitions = std::min(n, max_num_partitions_v);
  memr::async_device_resource resource{stream};

  // scalar_array
  memmg::managed_array<uint8_t> scalar_array{num_outputs * element_num_bytes * n, &resource};
  mtxb::make_device_scalar_array(scalar_array, stream, scalars, element_num_bytes, n);

  // bucket_count_array
  memmg::managed_array<unsigned> bucket_count_array{&resource};
  count_bucket_entries(bucket_count_array, stream, scalar_array, element_num_bytes, n, num_outputs,
                       bit_width, num_partitions);

  // bucket_count_sums
  memmg::managed_array<unsigned> bucket_count_sums{bucket_count_array.size()+1, &resource};
  algtr::exclusive_prefix_sum(bucket_count_sums, bucket_count_array, stream);

  // indexes
  memmg::managed_array<unsigned> index_count{1, memr::get_pinned_resource()};
  basdv::async_copy_device_to_host(
      index_count, basct::subspan(bucket_count_sums, bucket_count_array.size()), stream);
  co_await xendv::await_stream(stream);
  indexes.resize(index_count[0]);
  fill_index_kernel<<<num_partitions, dim3(num_outputs, num_bucket_groups, 1), 0, stream>>>(
      indexes.data(), bucket_count_sums.data(), scalar_array.data(), element_num_bytes, n,
      bit_width);
}

xena::future<>
fill_multiproduct_indexes(memmg::managed_array<bucket_descriptor>& bucket_descriptors,
                          memmg::managed_array<unsigned>& indexes, const basdv::stream& stream,
                          basct::cspan<const uint8_t*> scalars, unsigned element_num_bytes,
                          unsigned n, unsigned bit_width) noexcept {
  if (n == 0) {
    co_return;
  }
  auto num_outputs = scalars.size();
  auto num_bucket_groups = basn::divide_up(element_num_bytes * 8u, bit_width);
  auto num_partitions = std::min(n, max_num_partitions_v);
  memr::async_device_resource resource{stream};

  // scalar_array
  memmg::managed_array<uint8_t> scalar_array{num_outputs * element_num_bytes * n, &resource};
  mtxb::make_device_scalar_array(scalar_array, stream, scalars, element_num_bytes, n);

  // bucket_count_sums
  memmg::managed_array<unsigned> bucket_count_array{&resource};
  count_bucket_entries(bucket_count_array, stream, scalar_array, element_num_bytes, n, num_outputs,
                       bit_width, num_partitions);
  auto num_buckets = bucket_count_array.size();
  memmg::managed_array<unsigned> bucket_count_sums{num_buckets + 1, &resource};
  algtr::exclusive_prefix_sum(bucket_count_sums, bucket_count_array, stream);
  bucket_count_array.reset();

  // bucket_descriptors
  bucket_descriptors.resize(num_buckets);
  fill_bucket_descriptors_kernel<<<basn::divide_up(num_buckets, 256ul), 256u, 0, stream>>>(
      bucket_descriptors.data(), bucket_count_array.data(), num_partitions, num_buckets);

  // indexes
  memmg::managed_array<unsigned> index_count{1, memr::get_pinned_resource()};
  basdv::async_copy_device_to_host(
      index_count, basct::subspan(bucket_count_sums, bucket_count_array.size()), stream);
  co_await xendv::await_stream(stream);
  indexes.resize(index_count[0]);
  fill_index_kernel<<<num_partitions, dim3(num_outputs, num_bucket_groups, 1), 0, stream>>>(
      indexes.data(), bucket_count_sums.data(), scalar_array.data(), element_num_bytes, n,
      bit_width);
}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct_table
//--------------------------------------------------------------------------------------------------
xena::future<> compute_multiproduct_table(memmg::managed_array<bucket_descriptor>& table,
                                          memmg::managed_array<unsigned>& indexes,
                                          const basdv::stream& stream,
                                          basct::cspan<const uint8_t*> scalars,
                                          unsigned element_num_bytes, unsigned n,
                                          unsigned bit_width) noexcept {
  memr::async_device_resource resource{stream};

  // part 1
  memmg::managed_array<unsigned> bucket_counts{&resource};
  auto part1_fut = fill_multiproduct_indexes(bucket_counts, indexes, stream, scalars,
                                             element_num_bytes, n, bit_width);

  (void)fill_bucket_descriptors_kernel;
  // part 2
  co_await std::move(part1_fut);
/* struct bucket_descriptor { */
/*   unsigned num_entries; */
/*   unsigned bucket_index; */
/*   unsigned entry_first; */
/* }; */
}
} // namespace sxt::mtxbk
