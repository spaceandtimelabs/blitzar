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
