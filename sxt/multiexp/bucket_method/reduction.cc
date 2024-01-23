#include "sxt/multiexp/bucket_method/reduction.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// plan_reduction 
//--------------------------------------------------------------------------------------------------
unsigned plan_reduction(unsigned bit_width, unsigned num_buckets, unsigned num_outputs) noexcept {
  SXT_DEBUG_ASSERT(bit_width > 1u);
  auto num_buckets_per_group = (1u << bit_width) - 1u;
  auto num_bucket_groups = num_buckets / num_buckets_per_group;
  auto t = num_bucket_groups * num_outputs;
  if (t > 1024u || bit_width == 2u) {
    return bit_width - 1u; // 2
  }
  if (t > 512u || bit_width == 3u) {
    return bit_width - 2u; // 4
  }
  if (t > 256u || bit_width == 4u) {
    return bit_width - 3u; // 8
  }
  if (t > 128u || bit_width == 5u) {
    return bit_width - 4u; // 16
  }
  if (t > 64u || bit_width == 6u) {
    return bit_width - 5u; // 32
  }
  if (t > 32u || bit_width == 7u) {
    return bit_width - 6u; // 64
  }
  if (t > 16u || bit_width == 8u) {
    return bit_width - 7u; // 128
  }
  return bit_width - 8u; // 256
}
} // namespace sxt::mtxbk
