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

#include <cstddef>
#include <iterator>

#include "sxt/base/container/span.h"
#include "sxt/base/iterator/iterator_facade.h"

namespace sxt::basct {
//--------------------------------------------------------------------------------------------------
// span_iterator
//--------------------------------------------------------------------------------------------------
template <class T> class span_iterator final : public basit::iterator_facade<span_iterator<T>> {
public:
  using pointer = const span<T>*;
  using reference = const span<T>&;

  span_iterator() noexcept = default;

  span_iterator(T* data, size_t size) noexcept : data_{data}, size_{size} {}

  bool equal_to(span_iterator other) const noexcept { return data_ == other.data_; }

  span<T> dereference() const noexcept { return {data_, size_}; }

  void advance(ptrdiff_t delta) noexcept { data_ += delta * size_; }

  ptrdiff_t distance_to(span_iterator other) const noexcept {
    return std::distance(data_, other.data_) / size_;
  }

private:
  T* data_{nullptr};
  size_t size_{0};
};
} // namespace sxt::basct

namespace std {
template <class T>
struct iterator_traits<sxt::basct::span_iterator<T>>
    : sxt::basit::iterator_traits_impl<sxt::basct::span_iterator<T>> {};
} // namespace std
