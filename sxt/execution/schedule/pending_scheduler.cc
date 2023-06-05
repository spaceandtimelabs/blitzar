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
#include "sxt/execution/schedule/pending_scheduler.h"

#include <algorithm>

#include "sxt/base/error/assert.h"
#include "sxt/execution/schedule/pending_event.h"

namespace sxt::xens {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
pending_scheduler::pending_scheduler(size_t num_devices, size_t target_max_active) noexcept
    : target_max_active_{target_max_active}, active_counts_(num_devices) {}

//--------------------------------------------------------------------------------------------------
// on_event_new
//--------------------------------------------------------------------------------------------------
void pending_scheduler::on_event_new(int device) noexcept {
  SXT_DEBUG_ASSERT(0 <= device && static_cast<size_t>(device) < active_counts_.size());
  ++active_counts_[device];
}

//--------------------------------------------------------------------------------------------------
// on_event_done
//--------------------------------------------------------------------------------------------------
void pending_scheduler::on_event_done(int device) noexcept {
  SXT_DEBUG_ASSERT(0 <= device && static_cast<size_t>(device) < active_counts_.size());
  auto& count = active_counts_[device];
  SXT_DEBUG_ASSERT(count > 0);
  --count;
  while (head_ != nullptr && count < target_max_active_) {
    auto event = std::move(head_);
    head_ = event->release_next();
    event->invoke(device);
  }
}

//--------------------------------------------------------------------------------------------------
// schedule
//--------------------------------------------------------------------------------------------------
void pending_scheduler::schedule(std::unique_ptr<pending_event>&& event) noexcept {
  SXT_DEBUG_ASSERT(std::all_of(active_counts_.begin(), active_counts_.end(),
                               [&](size_t count) noexcept { return count >= target_max_active_; }));
  auto head = std::move(head_);
  head_ = std::move(event);
  head_->set_next(std::move(head));
}

//--------------------------------------------------------------------------------------------------
// get_available_device
//--------------------------------------------------------------------------------------------------
int pending_scheduler::get_available_device() const noexcept {
  if (active_counts_.empty()) {
    return -1;
  }
  int best_device = 0;
  size_t best_active_count = active_counts_[0];
  for (int device = 1; device < static_cast<int>(active_counts_.size()); ++device) {
    auto count = active_counts_[device];
    if (count < best_active_count) {
      best_device = device;
      best_active_count = count;
    }
  }
  if (best_active_count < target_max_active_) {
    return best_device;
  }
  return -1;
}
} // namespace sxt::xens
