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
#include "sxt/execution/schedule/scheduler.h"

#include "sxt/base/device/property.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/schedule/pollable_event.h"

namespace sxt::xens {
//--------------------------------------------------------------------------------------------------
// target_max_active_events_v
//--------------------------------------------------------------------------------------------------
static constexpr size_t target_max_active_events_v = 5;

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
scheduler::scheduler(size_t num_devices, size_t target_max_active) noexcept
    : pending_scheduler_{num_devices, target_max_active} {}

//--------------------------------------------------------------------------------------------------
// run
//--------------------------------------------------------------------------------------------------
void scheduler::run() noexcept {
  active_scheduler_.run([&](int device) noexcept { pending_scheduler_.on_event_done(device); });
}

//--------------------------------------------------------------------------------------------------
// schedule
//--------------------------------------------------------------------------------------------------
void scheduler::schedule(std::unique_ptr<pollable_event>&& event) noexcept {
  SXT_DEBUG_ASSERT(static_cast<size_t>(event->device()) < pending_scheduler_.num_devices());
  pending_scheduler_.on_event_new(event->device());
  active_scheduler_.schedule(std::move(event));
}

void scheduler::schedule(std::unique_ptr<pending_event>&& event) noexcept {
  pending_scheduler_.schedule(std::move(event));
}

//--------------------------------------------------------------------------------------------------
// get_available_device
//--------------------------------------------------------------------------------------------------
int scheduler::get_available_device() const noexcept {
  return pending_scheduler_.get_available_device();
}

//--------------------------------------------------------------------------------------------------
// get_scheduler
//--------------------------------------------------------------------------------------------------
scheduler& get_scheduler() noexcept {
  static thread_local auto instance =
      new scheduler{static_cast<size_t>(basdv::get_num_devices()), target_max_active_events_v};
  return *instance;
}
} // namespace sxt::xens
