#pragma once

#include <concepts>

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/state.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/type/value_type.h"
#include "sxt/execution/device/event_future.h"
#include "sxt/execution/device/synchronization.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// winked_device_copy
//--------------------------------------------------------------------------------------------------
template <class Cont, class T = bast::value_type_t<Cont>>
  requires std::convertible_to<Cont, basct::cspan<T>>
event_future<basct::span<T>> winked_device_copy(std::pmr::polymorphic_allocator<> alloc, const Cont& src) noexcept {
  auto res = basct::winked_span<T>(alloc, src.size());
  basdv::stream stream;
  basdv::async_copy_to_device(res, src, stream);
  basdv::event event;
  basdv::record_event(event, stream);
  computation_handle handle;
  handle.add_stream(std::move(stream));
  auto active_device = basdv::get_device();
  return event_future<basct::cspan<T>>{res, active_device, std::move(event), std::move(handle)};
}
} // namespace sxt::xendv
