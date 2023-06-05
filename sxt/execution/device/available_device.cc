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
#include "sxt/execution/device/available_device.h"

#include <memory>

#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/pending_event.h"
#include "sxt/execution/schedule/scheduler.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// available_device_awaiter
//--------------------------------------------------------------------------------------------------
namespace {
class available_device_awaiter final : public xens::pending_event {
public:
  available_device_awaiter(xena::promise<int>&& promise) noexcept : promise_{std::move(promise)} {}

  void invoke(int device) noexcept override { promise_.set_value(device); }

private:
  xena::promise<int> promise_;
};
} // namespace

//--------------------------------------------------------------------------------------------------
// await_available_device
//--------------------------------------------------------------------------------------------------
xena::future<int> await_available_device() noexcept {
  auto& scheduler = xens::get_scheduler();
  auto device = scheduler.get_available_device();
  if (device >= 0) {
    return xena::make_ready_future<int>(std::move(device));
  }
  xena::promise<int> promise;
  xena::future<int> res{promise};
  scheduler.schedule(std::make_unique<available_device_awaiter>(std::move(promise)));
  return res;
}
} // namespace sxt::xendv
