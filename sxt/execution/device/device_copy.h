/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
event_future<basct::span<T>> winked_device_copy(std::pmr::polymorphic_allocator<> alloc,
                                                const Cont& src) noexcept {
  if (src.empty()) {
    return event_future<basct::span<T>>{basct::span<T>{}};
  }
  auto res = basct::winked_span<T>(alloc, src.size());
  basdv::stream stream;
  basdv::async_copy_to_device(res, src, stream);
  basdv::event event;
  basdv::record_event(event, stream);
  computation_handle handle;
  handle.add_stream(std::move(stream));
  auto active_device = basdv::get_device();
  return event_future<basct::span<T>>{std::move(res), active_device, std::move(event),
                                      std::move(handle)};
}
} // namespace sxt::xendv
