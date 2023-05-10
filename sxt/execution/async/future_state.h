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
#pragma once

#include <concepts>
#include <optional>
#include <utility>

#include "sxt/base/error/assert.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// future_state
//--------------------------------------------------------------------------------------------------
template <class T> class future_state {
public:
  bool ready() const noexcept { return ready_; }

  const T& value() const noexcept {
    SXT_DEBUG_ASSERT(value_);
    return *value_;
  };

  T& value() noexcept { return *value_; }

  void make_ready() noexcept {
    SXT_DEBUG_ASSERT(value_, "value not set");
    ready_ = true;
  }

  template <class... Args>
    requires std::constructible_from<T, Args&&...>
  void emplace(Args&&... args) noexcept {
    SXT_DEBUG_ASSERT(!value_, "value already set");
    value_.emplace(std::forward<Args>(args)...);
  }

private:
  bool ready_{false};
  std::optional<T> value_;
};

template <> class future_state<void> {
public:
  bool ready() const noexcept { return ready_; }

  void make_ready() noexcept { ready_ = true; }

private:
  bool ready_{false};
};
} // namespace sxt::xena
