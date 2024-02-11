#include "sxt/multiexp/bucket_method/count.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// inclusive_prefix_count_buckets 
//--------------------------------------------------------------------------------------------------
void inclusive_prefix_count_buckets(basct::span<unsigned> counts, const basdv::stream& stream,
                                    basct::cspan<uint8_t> digits, unsigned element_num_bytes,
                                    unsigned bit_width, unsigned num_outputs, unsigned n,
                                    unsigned num_tiles) noexcept {
  auto num_buckets_per_digit = (1u << bit_width) - 1u;
  auto num_digits = basn::divide_up(element_num_bytes, 8u * bit_width);
  auto num_buckets_per_output = num_buckets_per_digit * num_digits;
  auto num_buckets_total = num_buckets_per_digit * num_outputs;
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
  (void)num_tiles;
}
} // namespace sxt::mtxbk
