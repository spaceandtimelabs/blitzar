#include "sxt/multiexp/bucket_method/count.h"

#include "cub/cub.cuh"

#include "sxt/algorithm/transform/prefix_sum.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// count_kernel 
//--------------------------------------------------------------------------------------------------
template <unsigned NumThreads, unsigned ItemsPerThread, unsigned NumBins>
static __global__ void count_kernel(unsigned* __restrict__ counts,
                                    const uint8_t* __restrict__ digits, unsigned n) noexcept {
  auto thread_index = threadIdx.x;
  auto tile_index = blockIdx.x;
  auto digit_index = blockIdx.y;
  auto output_index = blockIdx.z;

  auto num_tiles = gridDim.x;
  auto num_digits = gridDim.y;

  auto tile_size = NumThreads * ItemsPerThread;
  auto m = min(tile_size, n - tile_index * tile_size);

  // adjust pointers
  digits += output_index * num_digits * n;
  digits += n * digit_index;
  digits += tile_index * tile_size;

  counts += tile_index;
  counts += num_tiles * (NumBins - 1u) * digit_index;
  counts += num_tiles * (NumBins - 1u) * num_digits * output_index;

  // set up temporary workspace
  using BlockHistogram = cub::BlockHistogram<uint8_t, NumThreads, ItemsPerThread, NumBins>;
  __shared__ typename BlockHistogram::TempStorage temp_storage;
  __shared__ uint16_t bin_counts[NumBins];

  // load data
  uint8_t data[ItemsPerThread];
  for (unsigned i=0; i<ItemsPerThread; ++i) {
    auto index = thread_index + i * ItemsPerThread;
    if (index < m) {
      data[i] = digits[index];
    } else {
      data[i] = 0;
    }
  }

  // count
  BlockHistogram(temp_storage).InitHistogram(bin_counts);
  BlockHistogram(temp_storage).Histogram(data, bin_counts);

  // write results
  __syncthreads();
  for (unsigned i = thread_index; i < NumBins; i+=NumThreads) {
    // ignore zero counts
    if (i == 0) {
      continue;
    }
    counts[(i - 1u) * num_tiles] = bin_counts[i];
  }
}

//--------------------------------------------------------------------------------------------------
// count_kernel 
//--------------------------------------------------------------------------------------------------
template <unsigned NumThreads, unsigned ItemsPerThread, unsigned NumBins>
static __global__ void count_kernel(unsigned* __restrict__ counts,
                                    uint16_t* __restrict__ tile_counts,
                                    const uint8_t* __restrict__ digits, unsigned n) noexcept {
  auto thread_index = threadIdx.x;
  auto tile_index = blockIdx.x;
  auto digit_index = blockIdx.y;
  auto output_index = blockIdx.z;

  auto num_tiles = gridDim.x;
  auto num_digits = gridDim.y;

  auto tile_size = NumThreads * ItemsPerThread;
  auto m = min(tile_size, n - tile_index * tile_size);

  // adjust pointers
  digits += output_index * num_digits * n;
  digits += n * digit_index;
  digits += tile_index * tile_size;

  counts += tile_index;
  counts += num_tiles * (NumBins - 1u) * digit_index;
  counts += num_tiles * (NumBins - 1u) * num_digits * output_index;

  // set up temporary workspace
  static_assert(NumBins % NumThreads == 0);
  constexpr auto prefix_counts_items_per_thread = NumBins / NumThreads;
  using BlockHistogram = cub::BlockHistogram<uint8_t, NumThreads, ItemsPerThread, NumBins>;
  using BlockScan = cub::BlockScan<uint16_t, NumThreads>;

  __shared__ union {
    typename BlockHistogram::TempStorage histogram;
    typename BlockScan::TempStorage scan;
  } temp_storage;

  __shared__ uint16_t bin_counts[NumBins];

  // load data
  uint8_t data[ItemsPerThread];
  for (unsigned i=0; i<ItemsPerThread; ++i) {
    auto index = thread_index + i * ItemsPerThread;
    if (index < m) {
      data[i] = digits[index];
    } else {
      data[i] = 0;
    }
  }

  // count
  BlockHistogram(temp_storage.histogram).InitHistogram(bin_counts);
  BlockHistogram(temp_storage.histogram).Histogram(data, bin_counts);

  // compute prefix counts
  __syncthreads();
  (void)tile_counts;
  uint16_t prefix_sums[prefix_counts_items_per_thread];
  for (unsigned i=0; i<prefix_counts_items_per_thread; ++i) {
    auto index = thread_index * prefix_counts_items_per_thread + i;
    prefix_sums[i] = bin_counts[index] * (index != 0);
  }
  BlockScan(temp_storage.scan).InclusiveSum(prefix_sums, prefix_sums);

  // write count results
  __syncthreads();
  for (unsigned i = thread_index; i < NumBins; i+=NumThreads) {
    // ignore zero counts
    if (i == 0) {
      continue;
    }
    counts[(i - 1u) * num_tiles] = bin_counts[i];
  }
}

//--------------------------------------------------------------------------------------------------
// inclusive_prefix_count_buckets 
//--------------------------------------------------------------------------------------------------
void inclusive_prefix_count_buckets(basct::span<unsigned> counts, const basdv::stream& stream,
                                    basct::cspan<uint8_t> digits, unsigned element_num_bytes,
                                    unsigned bit_width, unsigned num_outputs, unsigned tile_size,
                                    unsigned n) noexcept {
  auto num_buckets_per_digit = (1u << bit_width) - 1u;
  auto num_digits = basn::divide_up(element_num_bytes, 8u * bit_width);
  auto num_buckets_per_output = num_buckets_per_digit * num_digits;
  auto num_buckets_total = num_buckets_per_digit * num_outputs;
  auto num_tiles = basn::divide_up(n, tile_size);
  SXT_DEBUG_ASSERT(
      // clang-format off
      counts.size() == num_buckets_total * num_tiles &&
      digits.size() == num_digits * n * num_outputs &&
      basdv::is_active_device_pointer(counts.data()) &&
      basdv::is_active_device_pointer(digits.data())
      // clang-format on
  );
  SXT_RELEASE_ASSERT(bit_width == 8 && tile_size == 1024,
                     "only support these values for now");
  memr::async_device_resource resource{stream};
  memmg::managed_array<unsigned> counts_p{counts.size(), &resource};

  static constexpr unsigned num_threads = 128;
  static constexpr unsigned items_per_thread = 8;
  SXT_DEBUG_ASSERT(num_threads * items_per_thread == tile_size);
  // count
  count_kernel<num_threads, items_per_thread, 256>
      <<<dim3(num_tiles, num_digits, num_outputs), num_threads, 0, stream>>>(counts_p.data(),
                                                                             digits.data(), n);

  // compute prefix sums
  algtr::inclusive_prefix_sum(counts, counts_p, stream);
}

void inclusive_prefix_count_buckets(basct::span<unsigned> counts, basct::span<uint16_t> tile_counts,
                                    const basdv::stream& stream, basct::cspan<uint8_t> digits,
                                    unsigned element_num_bytes, unsigned bit_width,
                                    unsigned num_outputs, unsigned tile_size, unsigned n) noexcept {
  auto num_buckets_per_digit = (1u << bit_width) - 1u;
  auto num_digits = basn::divide_up(element_num_bytes, 8u * bit_width);
  auto num_buckets_per_output = num_buckets_per_digit * num_digits;
  auto num_buckets_total = num_buckets_per_digit * num_outputs;
  auto num_tiles = basn::divide_up(n, tile_size);
  SXT_DEBUG_ASSERT(
      // clang-format off
      counts.size() == tile_counts.size() &&
      counts.size() == num_buckets_total * num_tiles &&
      digits.size() == num_digits * n * num_outputs &&
      basdv::is_active_device_pointer(counts.data()) &&
      basdv::is_active_device_pointer(digits.data())
      // clang-format on
  );
  SXT_RELEASE_ASSERT(bit_width == 8 && tile_size == 1024,
                     "only support these values for now");
  memr::async_device_resource resource{stream};
  memmg::managed_array<unsigned> counts_p{counts.size(), &resource};

  // count
  static constexpr unsigned num_threads = 128;
  static constexpr unsigned items_per_thread = 8;
  SXT_DEBUG_ASSERT(num_threads * items_per_thread == tile_size);
  count_kernel<num_threads, items_per_thread, 256>
      <<<dim3(num_tiles, num_digits, num_outputs), num_threads, 0, stream>>>(
          counts_p.data(), tile_counts.data(), digits.data(), n);

  // compute prefix sums
  algtr::inclusive_prefix_sum(counts, counts_p, stream);
}
} // namespace sxt::mtxbk
