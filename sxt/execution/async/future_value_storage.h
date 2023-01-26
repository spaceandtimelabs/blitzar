#pragma once

#include <utility>

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// future_value_storage
//--------------------------------------------------------------------------------------------------
template <class T> class future_value_storage {
public:
  future_value_storage() noexcept = default;

  explicit future_value_storage(T&& value) noexcept : value_{std::move(value)} {}

  operator T&() noexcept { return value_; }

  T consume_value() noexcept { return T{std::move(value_)}; }

private:
  T value_;
};

template <> class future_value_storage<void> {
public:
  future_value_storage() noexcept = default;

  void consume_value() noexcept {}
};
} // namespace sxt::xena
