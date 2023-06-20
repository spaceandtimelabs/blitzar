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

#include <algorithm>

#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/iterator_facade.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/base/type/narrow_cast.h"

namespace sxt::basit {
//--------------------------------------------------------------------------------------------------
// index_range_iterator
//--------------------------------------------------------------------------------------------------
class index_range_iterator final : public iterator_facade<index_range_iterator> {
public:
  using pointer = const int*;
  using reference = const int&;

  index_range_iterator() noexcept = default;

  index_range_iterator(const index_range& range, size_t step) noexcept
      : range_{range}, step_{step} {}

  bool equal_to(const index_range_iterator& other) const noexcept {
    return range_ == other.range_ && step_ == other.step_;
  }

  index_range dereference() const noexcept {
    auto a = this->range_.a();
    auto b = std::min(a + step_, this->range_.b());
    return index_range{a, b};
  }

  void advance(ptrdiff_t delta) noexcept {
    auto i = range_.a() + step_ * delta;
    SXT_DEBUG_ASSERT(i >= 0);
    auto ap = bast::narrow_cast<size_t>(std::min(i, range_.b()));
    range_ = index_range{ap, range_.b()};
  }

  ptrdiff_t distance_to(const index_range_iterator& other) const noexcept {
    // clang-format off
    SXT_DEBUG_ASSERT(
        other.range_.b() == range_.b() && 
        step_ == other.step_ && 
        step_ != 0
    );
    // clang-format on
    auto delta = other.range_.a() - range_.a();
    return basn::divide_up(delta, step_);
  }

private:
  index_range range_;
  size_t step_{0};
};
} // namespace sxt::basit

namespace std {
template <>
struct iterator_traits<sxt::basit::index_range_iterator>
    : sxt::basit::iterator_traits_impl<sxt::basit::index_range_iterator> {};
} // namespace std
