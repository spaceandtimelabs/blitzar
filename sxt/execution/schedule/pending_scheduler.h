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

#include <memory>
#include <vector>

#include "sxt/execution/schedule/pending_event.h"

namespace sxt::xens {
//--------------------------------------------------------------------------------------------------
// pending_scheduler
//--------------------------------------------------------------------------------------------------
class pending_scheduler {
public:
  pending_scheduler(size_t num_devices, size_t target_max_active) noexcept;

  void on_event_new(int device) noexcept;

  void on_event_done(int device) noexcept;

  void schedule(std::unique_ptr<pending_event>&& event) noexcept;

  int get_available_device() const noexcept;

  size_t num_devices() const noexcept { return active_counts_.size(); }

private:
  size_t target_max_active_;
  std::vector<size_t> active_counts_;
  std::unique_ptr<pending_event> head_;
};
} // namespace sxt::xens
