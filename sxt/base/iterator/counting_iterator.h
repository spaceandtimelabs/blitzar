#pragma once

#include <type_traits>

#include "sxt/base/iterator/iterator_facade.h"

namespace sxt::basit {
//--------------------------------------------------------------------------------------------------
// counting_iterator
//--------------------------------------------------------------------------------------------------
template <class T> class counting_iterator final : public iterator_facade<counting_iterator<T>> {
  static_assert(std::is_integral_v<T>);

public:
  using pointer = const T*;
  using reference = const T&;

  counting_iterator() noexcept = default;

  explicit counting_iterator(T counter) noexcept : counter_{counter} {}

  bool equal_to(counting_iterator other) const noexcept { return counter_ == other.counter_; }

  T dereference() const noexcept { return this->counter_; }

  void advance(ptrdiff_t delta) noexcept { counter_ += delta; }

  int distance_to(counting_iterator other) const noexcept { return other.counter_ - counter_; }

private:
  T counter_{0};
};
} // namespace sxt::basit

namespace std {
template <class T>
struct iterator_traits<sxt::basit::counting_iterator<T>>
    : sxt::basit::iterator_traits_impl<sxt::basit::counting_iterator<T>> {};
} // namespace std
