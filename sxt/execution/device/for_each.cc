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
#include "sxt/execution/device/for_each.h"

#include "sxt/base/device/active_device_guard.h"
#include "sxt/base/device/property.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/available_device.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// concurrent_for_each
//--------------------------------------------------------------------------------------------------
xena::future<>
concurrent_for_each(basit::index_range rng,
                    std::function<xena::future<>(const basit::index_range&)> f) noexcept {
  auto [it, last] = basit::split(rng, basdv::get_num_devices());
  std::vector<xena::future<>> futs;
  futs.reserve(std::distance(it, last));
  for (; it != last; ++it) {
    auto device = co_await await_available_device();
    basdv::active_device_guard guard{device};
    futs.emplace_back(f(*it));
  }
  for (auto& fut : futs) {
    co_await std::move(fut);
  }
}
} // namespace sxt::xendv
