#include "sxt/multiexp/bucket_method/multiproduct_table.h"

#include "cub/cub.cuh"

#include "sxt/base/device/stream.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/base/scalar_array.h"
#include "sxt/multiexp/bucket_method/count.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// multiproduct_table_kernel 
//--------------------------------------------------------------------------------------------------
template <uint16_t NumThreads, uint16_t ItemsPerThread>
static __global__ void multiproduct_table_kernel(uint16_t* __restrict__ bucket_prefix_counts,
                                                 uint16_t* __restrict__ indexes,
                                                 const uint8_t* __restrict__ bytes,
                                                 unsigned bit_width, unsigned n) {
  uint16_t thread_index = threadIdx.x;
  auto digit_index = blockDim.x;
  auto num_digits = gridDim.x;
  auto output_index = blockDim.y;
  auto num_buckets_per_digit = (1u << bit_width) - 1u;

  // algorithms and shared memory
  using RadixSort = cub::BlockRadixSort<uint8_t, NumThreads, ItemsPerThread, uint16_t>;
  __shared__ union {
    RadixSort::TempStorage sort;
  } temp_storage;

  // adjust pointers
  bucket_prefix_counts += digit_index * num_buckets_per_digit;
  bucket_prefix_counts += output_index * num_digits * num_buckets_per_digit;
  indexes += digit_index * n;
  indexes += output_index * num_digits * n;
  bytes += digit_index * n;
  bytes += output_index * num_digits * n;

  // load bytes
  uint8_t keys[ItemsPerThread];
  uint16_t values[ItemsPerThread];
  for (uint16_t i=0; i<ItemsPerThread; ++i) {
    auto index = thread_index + i * NumThreads;
    if (index < n) {
      keys[i] = bytes[index];
      values[i] = index;
    } else {
      keys[i] = 0;
      values[i] = 0;
    }
  }

  // sort
  RadixSort(temp_storage.sort).Sort(keys, values);

  // determine bucket counts
  // prefix sum buckets
}

//--------------------------------------------------------------------------------------------------
// make_multiproduct_table 
//--------------------------------------------------------------------------------------------------
xena::future<> make_multiproduct_table(basct::span<unsigned> bucket_prefix_counts,
                                       memmg::managed_array<unsigned>& indexes,
                                       basct::cspan<const uint8_t*> scalars,
                                       unsigned element_num_bytes, unsigned bit_width,
                                       unsigned n) noexcept {
  auto num_outputs = scalars.size();
  auto num_buckets_per_digit = (1u << bit_width) - 1u;
  auto num_digits = basn::divide_up(element_num_bytes * 8u, bit_width);
  auto num_buckets_per_output = num_buckets_per_digit * num_digits;
  auto num_buckets_total = num_buckets_per_output * num_outputs;
  static constexpr unsigned tile_size = 1024;
  auto num_tiles = basn::divide_up(n, tile_size);

  // transpose scalars
  memmg::managed_array<uint8_t> bytes{num_outputs * n * element_num_bytes,
                                      memr::get_device_resource()};
  co_await mtxb::transpose_scalars_to_device(bytes, scalars, element_num_bytes, bit_width, n);

  // count bucket entries
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<unsigned> partial_prefix_counts{num_buckets_total * num_tiles, &resource};
  memmg::managed_array<uint16_t> partial_prefix_tile_counts{num_buckets_total * num_tiles, &resource};
  inclusive_prefix_count_buckets(partial_prefix_counts, partial_prefix_tile_counts, stream, bytes,
                                 element_num_bytes, bit_width, num_outputs, tile_size, n);

  // fill in bucket prefix counts

  // fill in multiproduct table
  (void)stream;
  /* void inclusive_prefix_count_buckets(basct::span<unsigned> counts, basct::span<uint16_t>
   * tile_counts, */
  /*                                     const basdv::stream& stream, basct::cspan<uint8_t> digits,
   */
  /*                                     unsigned element_num_bytes, unsigned bit_width, */
  /*                                     unsigned num_outputs, unsigned tile_size, unsigned n)
   * noexcept; */
  (void)bucket_prefix_counts;
  (void)indexes;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
  (void)n;
}

xena::future<> make_multiproduct_table(basct::span<uint16_t> bucket_prefix_counts,
                                       basct::span<uint16_t> indexes,
                                       basct::cspan<const uint8_t*> scalars,
                                       unsigned element_num_bytes, unsigned bit_width,
                                       unsigned n) noexcept {
  auto num_outputs = scalars.size();
  auto num_buckets_per_digit = (1u << bit_width) - 1u;
  auto num_digits = basn::divide_up(element_num_bytes * 8u, bit_width);
  auto num_buckets_per_output = num_buckets_per_digit * num_digits;
  auto num_buckets_total = num_buckets_per_output * num_outputs;

  // transpose scalars
  memmg::managed_array<uint8_t> bytes{num_outputs * n * element_num_bytes,
                                      memr::get_device_resource()};
  co_await mtxb::transpose_scalars_to_device(bytes, scalars, element_num_bytes, bit_width, n);

  // compute buckets
  (void)num_buckets_total;
  (void)bucket_prefix_counts;
  (void)indexes;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
  (void)n;
}
} // namespace sxt::mtxbk
