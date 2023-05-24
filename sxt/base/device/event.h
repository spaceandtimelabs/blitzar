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

#include "sxt/base/type/raw_cuda_event.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// event
//--------------------------------------------------------------------------------------------------
class event {
public:
  event() noexcept;
  event(const event&) = delete;
  event(event&& other) noexcept;

  ~event() noexcept;

  event& operator=(const event&) = delete;
  event& operator=(event&& other) noexcept;

  operator const CUevent_st*() const noexcept { return event_; }

  operator bast::raw_cuda_event_t() noexcept { return event_; }

  void clear() noexcept;

  bool query_is_ready() noexcept;

private:
  bast::raw_cuda_event_t event_;
};
} // namespace sxt::basdv
