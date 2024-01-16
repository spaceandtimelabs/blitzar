#include "sxt/multiexp/bucket_method/multiproduct_table.h"

#include "sxt/base/device/stream.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/multiexp/base/scalar_array.h"
#include "sxt/multiexp/bucket_method/count.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// compute_multiproduct_table
//--------------------------------------------------------------------------------------------------
xena::future<void> compute_multiproduct_table(memmg::managed_array<bucket_descriptor>& table,
                                              memmg::managed_array<unsigned>& indexes,
                                              const basdv::stream& stream,
                                              basct::cspan<const uint8_t*> scalars,
                                              unsigned element_num_bytes, unsigned n,
                                              unsigned bit_width) noexcept {
  auto num_outputs = scalars.size();
  const unsigned num_partitions = 64;
  memr::async_device_resource resource{stream};

  // scalar_array
  memmg::managed_array<uint8_t> scalar_array{num_outputs * element_num_bytes * n, &resource};
  mtxb::make_device_scalar_array(scalar_array, stream, scalars, element_num_bytes, n);

  // bucket_count_array
  memmg::managed_array<unsigned> bucket_count_array{&resource};
  count_bucket_entries(bucket_count_array, stream, scalar_array, element_num_bytes, n, num_outputs,
                       bit_width, num_partitions);

  (void)table;
  (void)indexes;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
}
} // namespace sxt::mtxbk
