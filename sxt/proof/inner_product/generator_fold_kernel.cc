#include "sxt/proof/inner_product/generator_fold_kernel.h"

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/proof/inner_product/generator_fold.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// fold_generators
//--------------------------------------------------------------------------------------------------
xena::future<void> fold_generators(basct::span<c21t::element_p3> g_vector,
                                   basct::cspan<unsigned> decomposition) noexcept {
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_device_pointer(g_vector.data()) &&
      basdv::is_host_pointer(decomposition.data()) &&
      g_vector.size() % 2 == 0 && 
      g_vector.size() > 1
      // clang-format on
  );
  auto n = static_cast<unsigned>(g_vector.size() / 2);
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<unsigned> decomposition_gpu{decomposition.size(), &resource};
  basdv::async_copy_host_to_device(decomposition_gpu, decomposition, stream);
  auto data = g_vector.data();
  auto decomposition_data = decomposition_gpu.data();
  auto decomposition_size = static_cast<unsigned>(decomposition.size());
  auto f = [
               // clang-format off
    data,
    decomposition_data,
    decomposition_size
               // clang-format on
  ] __device__
           __host__(unsigned n, unsigned i) noexcept {
             fold_generators(data[i],
                             basct::cspan<unsigned>{decomposition_data, decomposition_size},
                             data[i], data[i + n]);
           };
  return algi::for_each(std::move(stream), f, n);
}
} // namespace sxt::prfip
