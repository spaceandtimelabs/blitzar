/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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

#include <immintrin.h>

#include "sxt/execution/schedule/pollable_event.h"

namespace sxt::xens {
//--------------------------------------------------------------------------------------------------
// run
//--------------------------------------------------------------------------------------------------
void scheduler::run() noexcept {
  // Simple loop to poll and execute ready tasks
  while (true) {
    if (head_ == nullptr) {
      return;
    }
    // Note: this code hasn't been informed by benchmarking yet, and
    // it's likely that it will be adjusted when it does.
    //
    // For now, we emit a pause instruction as that frequently leads to
    // better performance for polling code.
    //
    // See https://stackoverflow.com/q/58424276/4447365
    _mm_pause();

    if (head_->ready()) {
      auto event = std::move(head_);
      head_ = event->release_next();
      event->invoke();
      continue;
    }
    auto event = head_.get();
    while (event->next() != nullptr) {
      auto next = event->next();
      if (next->ready()) {
        next->invoke();
        event->set_next(next->release_next());
      } else {
        event = next;
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
// schedule
//--------------------------------------------------------------------------------------------------
void scheduler::schedule(std::unique_ptr<pollable_event>&& event) noexcept {
  auto head = std::move(head_);
  head_ = std::move(event);
  head_->set_next(std::move(head));
}

//--------------------------------------------------------------------------------------------------
// get_scheduler
//--------------------------------------------------------------------------------------------------
scheduler& get_scheduler() noexcept {
  static thread_local auto instance = new scheduler{};
  return *instance;
}
} // namespace sxt::xens
