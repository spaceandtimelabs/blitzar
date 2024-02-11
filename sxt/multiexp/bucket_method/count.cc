#include "sxt/multiexp/bucket_method/count.h"

#include "cub/cub.cuh"

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
      data[i] = digits[i];
    } else {
      data[i] = 0;
    }
  }

  // count
  BlockHistogram(temp_storage).Histogram(data, bin_counts);

  // write results
  __syncthreads();
  for (unsigned i = thread_index; i < NumBins; i+=NumThreads) {
    // ignore zero counts
    if (i == 0) {
      continue;
    }
    counts[i * tile_size] = bin_counts[i];
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
  memr::async_device_resource resource{stream};
  memmg::managed_array<unsigned> counts_p{counts.size(), &resource};

  // count
  count_kernel<256, 4, 256>
      <<<dim3(num_tiles, num_digits, num_outputs), 256>>>(counts_p.data(), digits.data(), n);
  /* template <unsigned NumThreads, unsigned ItemsPerThread, unsigned NumBins> */
  /* static __global__ void count_kernel(unsigned* __restrict__ counts, */
  /*                                     const uint8_t* __restrict__ digits, unsigned n) noexcept {
   */
  
  // compute prefix sums
  (void)counts;
  (void)stream;
  (void)digits;
  (void)element_num_bytes;
  (void)bit_width;
  (void)num_outputs;
  (void)n;
  (void)tile_size;
}
} // namespace sxt::mtxbk
