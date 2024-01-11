#include "sxt/multiexp/bucket_method/count.h"

#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"

namespace sxt::mtxbk {

//--------------------------------------------------------------------------------------------------
// count_bucket_entries
//--------------------------------------------------------------------------------------------------
xena::future<> count_bucket_entries(memmg::managed_array<unsigned>& count_array,
                                    basct::cspan<uint8_t> scalars, unsigned element_num_bytes,
                                    unsigned bit_width, unsigned num_partitions) noexcept {
  auto num_bucket_groups = basn::divide_up(element_num_bytes * 8u, bit_width);
  auto num_outputs = scalars.size();
  auto num_buckets_per_group = 1u << bit_width;
  count_array.resize(num_outputs * num_bucket_groups * num_buckets_per_group * num_partitions);
  (void)num_outputs;
  (void)num_bucket_groups;
  (void)count_array;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
  (void)num_partitions;
  return {};
}
} // namespace sxt::mtxbk
