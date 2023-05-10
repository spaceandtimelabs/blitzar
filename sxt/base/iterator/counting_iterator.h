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

  ptrdiff_t distance_to(counting_iterator other) const noexcept {
    return other.counter_ - counter_;
  }

private:
  T counter_{0};
};
} // namespace sxt::basit

namespace std {
template <class T>
struct iterator_traits<sxt::basit::counting_iterator<T>>
    : sxt::basit::iterator_traits_impl<sxt::basit::counting_iterator<T>> {};
} // namespace std
