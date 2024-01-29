#include "sxt/multiexp/base/scalar_array.h"

#include <vector>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// make_transposed_device_scalar_array_impl
//--------------------------------------------------------------------------------------------------
static xena::future<> make_transposed_device_scalar_array_impl(basct::span<uint8_t> array_slice,
                                                               basct::cspan<uint8_t> scalars,
                                                               unsigned element_num_bytes,
                                                               unsigned n) noexcept {
  (void)array_slice;
  (void)scalars;
  (void)element_num_bytes;
  (void)n;
  return {};
}

//--------------------------------------------------------------------------------------------------
// make_device_scalar_array 
//--------------------------------------------------------------------------------------------------
void make_device_scalar_array(basct::span<uint8_t> array, const basdv::stream& stream,
                              basct::cspan<const uint8_t*> scalars, size_t element_num_bytes,
                              size_t n) noexcept {
  auto num_outputs = scalars.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      array.size() == num_outputs * element_num_bytes * n &&
      basdv::is_active_device_pointer(array.data())
      // clang-format on
  );
  auto num_bytes_per_output = element_num_bytes * n;
  for (size_t output_index = 0; output_index<num_outputs; ++output_index) {
    auto output_first = output_index * element_num_bytes * n;
    SXT_DEBUG_ASSERT(basdv::is_host_pointer(scalars[output_index]));
    basdv::async_copy_host_to_device(
        array.subspan(output_first, num_bytes_per_output),
        basct::cspan<uint8_t>{scalars[output_index], num_bytes_per_output}, stream);
  }
}

//--------------------------------------------------------------------------------------------------
// make_transposed_device_scalar_array 
//--------------------------------------------------------------------------------------------------
xena::future<> make_transposed_device_scalar_array(basct::span<uint8_t> array,
                                                   basct::cspan<const uint8_t*> scalars,
                                                   unsigned element_num_bytes,
                                                   unsigned n) noexcept {
  auto num_outputs = scalars.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      array.size() == num_outputs * element_num_bytes * n &&
      basdv::is_active_device_pointer(array.data())
      // clang-format on
  );
  std::vector<xena::future<>> futs(num_outputs);
  for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
    futs[output_index] = make_transposed_device_scalar_array_impl(
        array.subspan(output_index * element_num_bytes * n, element_num_bytes * n),
        basct::cspan<uint8_t>{scalars[output_index], n * element_num_bytes}, element_num_bytes, n);
  }
  for (auto& fut : futs) {
    co_await std::move(fut);
  }
}
} // namespace sxt::mtxb
