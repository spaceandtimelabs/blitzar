#include "sxt/proof/inner_product/gpu_workspace.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/synchronization.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
gpu_workspace::gpu_workspace() noexcept
    : a_vector{memr::get_device_resource()}, b_vector{memr::get_device_resource()},
      g_vector{memr::get_device_resource()} {}

//--------------------------------------------------------------------------------------------------
// ap_value
//--------------------------------------------------------------------------------------------------
xena::future<> gpu_workspace::ap_value(s25t::element& value) const noexcept {
  SXT_DEBUG_ASSERT(this->a_vector.size() == 1);
  memmg::managed_array<s25t::element> value_p{1, memr::get_pinned_resource()};
  basdv::stream stream;
  basdv::async_copy_device_to_host(value_p, this->a_vector, stream);
  co_await xena::await_and_own_stream(std::move(stream));
  value = value_p[0];
}
} // namespace sxt::prfip
