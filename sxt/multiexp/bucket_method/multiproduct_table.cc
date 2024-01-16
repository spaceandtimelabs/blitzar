#include "sxt/multiexp/bucket_method/multiproduct_table.h"

#include "sxt/algorithm/transform/prefix_sum.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/scalar_array.h"
#include "sxt/multiexp/bucket_method/count.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// max_num_partitions_v
//--------------------------------------------------------------------------------------------------
const unsigned max_num_partitions_v = 64;

//--------------------------------------------------------------------------------------------------
// index_kernel
//--------------------------------------------------------------------------------------------------
static __global__ void index_kernel(unsigned* indexes, const unsigned* count_sums,
                                    const uint8_t* scalars, unsigned element_num_bytes,
                                    unsigned bit_width, unsigned n) {
  (void)indexes;
  (void)count_sums;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
  (void)n;
#if 0
  unsigned output_index = threadIdx.x;
  unsigned bucket_group_index = threadIdx.y;
  unsigned partition_index = blockIdx.x;
  unsigned num_partitions = gridDim.x;
  unsigned num_buckets_per_group = (1u << bit_width) - 1;
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
#endif
}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct_table_part1 
//--------------------------------------------------------------------------------------------------
xena::future<> compute_multiproduct_table_part1(memmg::managed_array<unsigned>& bucket_counts,
                                                memmg::managed_array<unsigned>& indexes,
                                                const basdv::stream& stream,
                                                basct::cspan<const uint8_t*> scalars,
                                                unsigned element_num_bytes, unsigned n,
                                                unsigned bit_width) noexcept {
  auto num_outputs = scalars.size();
  memr::async_device_resource resource{stream};

  // scalar_array
  memmg::managed_array<uint8_t> scalar_array{num_outputs * element_num_bytes * n, &resource};
  mtxb::make_device_scalar_array(scalar_array, stream, scalars, element_num_bytes, n);

  // bucket_count_array
  memmg::managed_array<unsigned> bucket_count_array{&resource};
  count_bucket_entries(bucket_count_array, stream, scalar_array, element_num_bytes, n, num_outputs,
                       bit_width, max_num_partitions_v);

  // bucket_count_sums
  memmg::managed_array<unsigned> bucket_count_sums{bucket_count_array.size()+1, &resource};
  algtr::exclusive_prefix_sum(bucket_count_sums, bucket_count_array, stream);

  // indexes
  memmg::managed_array<unsigned> index_count{1, memr::get_pinned_resource()};
  basdv::async_copy_device_to_host(
      index_count, basct::subspan(bucket_count_sums, bucket_count_array.size()), stream);
  (void)index_kernel;
  /* index_kernel<<<num_partitions, dim3(num_outputs, num_bucket_groups, 1), 0, stream>>>( */
  /*     count_array.data(), scalars.data(), element_num_bytes, bit_width, n); */
  (void)bucket_counts;
  (void)indexes;
  (void)stream;
  (void)scalars;
  (void)element_num_bytes;
  (void)n;
  (void)bit_width;
  return {};
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
  auto num_outputs = scalars.size();
  memr::async_device_resource resource{stream};

  // part 1
  memmg::managed_array<unsigned> bucket_counts{&resource};
  auto part1_fut = compute_multiproduct_table_part1(bucket_counts, indexes, stream, scalars,
                                                    element_num_bytes, n, bit_width);

  // part 2
  co_await std::move(part1_fut);
}
} // namespace sxt::mtxbk
