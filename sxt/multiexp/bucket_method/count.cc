#include "sxt/multiexp/bucket_method/count.h"

#include "cub/cub.cuh"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// count_kernel 
//--------------------------------------------------------------------------------------------------
static __global__ void count_kernel(unsigned* __restrict__ counts,
                                    const uint8_t* __restrict__ digits, unsigned n) noexcept {
  auto tile_index = blockIdx.x;
  auto digit_index = blockIdx.y;
  auto output_index = blockIdx.z;

  auto num_tiles = gridDim.x;
  auto num_digits = gridDim.y;
  auto tile_size = basn::divide_up(n, num_tiles);

  auto cnt = min(n - tile_index * tile_size, tile_size);

  // adjust pointers
  digits += output_index * num_digits * n;
  digits += n * digit_index;
  digits += tile_index * tile_size;

  // count
  (void)cnt;
  // prefix sum counts
  (void)tile_index;
  (void)digit_index;
  (void)output_index;

  (void)counts;
  (void)digits;
  (void)n;
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
