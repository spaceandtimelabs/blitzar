#include "sxt/multiexp/bucket_method/multiproduct_table.h"

#include "cub/cub.cuh"

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/log/log.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/base/scalar_array.h"
#include "sxt/multiexp/bucket_method/multiproduct_table_kernel.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// make_multiproduct_table 
//--------------------------------------------------------------------------------------------------
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
  SXT_DEBUG_ASSERT(
      bucket_prefix_counts.size() == num_buckets_total &&
      indexes.size() == num_outputs * num_digits * n &&
      basdv::is_active_device_pointer(bucket_prefix_counts.data()) &&
      basdv::is_active_device_pointer(indexes.data())
  );

  // transpose scalars
  basl::info("copying scalars to device");
  memmg::managed_array<uint8_t> bytes{num_outputs * n * element_num_bytes,
                                      memr::get_device_resource()};
  co_await mtxb::transpose_scalars_to_device(bytes, scalars, element_num_bytes, bit_width, n);

  // compute buckets
  basl::info("computing multiproduct decomposition");
  SXT_RELEASE_ASSERT(bit_width == 8u, "only support bit_width == 8u for now");
  SXT_RELEASE_ASSERT(n <= 1024, "only support n <= 1024 for now");
  static constexpr unsigned num_threads = 128;
  static constexpr unsigned items_per_thread = 8;
  basdv::stream stream;
  multiproduct_table_kernel<num_threads, items_per_thread, 8>
      <<<dim3(num_digits, num_outputs, 1), num_threads, 0, stream>>>(
          bucket_prefix_counts.data(), indexes.data(), bytes.data(), n);

  // prefix sum
  auto f = [bucket_prefix_counts = bucket_prefix_counts.data(),
            num_buckets_per_digit = num_buckets_per_digit] __host__
           __device__(unsigned /*num_digits_total*/, unsigned index) noexcept {
             auto counts = bucket_prefix_counts + index * num_buckets_per_digit;
             for (unsigned i = 1; i < num_buckets_per_digit; ++i) {
               counts[i] += counts[i - 1u];
             }
           };
  algi::launch_for_each_kernel(stream, f, num_digits * num_outputs);
  co_await xendv::await_stream(stream);
}
} // namespace sxt::mtxbk
