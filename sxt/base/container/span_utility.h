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
#include <cstddef>
#include <type_traits>

#include "sxt/base/container/span.h"

namespace sxt::basct {
//--------------------------------------------------------------------------------------------------
// subspan
//--------------------------------------------------------------------------------------------------
template <class Cont, class T = std::remove_pointer_t<decltype(std::declval<Cont>().data())>>
span<T> subspan(Cont&& cont, size_t offset) noexcept
  requires requires {
    { cont.data() } -> std::convertible_to<T*>;
    { cont.size() } -> std::convertible_to<size_t>;
  }
{
  return span<T>{cont}.subspan(offset);
}

template <class Cont, class T = std::remove_pointer_t<decltype(std::declval<Cont>().data())>>
span<T> subspan(Cont&& cont, size_t offset, size_t size) noexcept
  requires requires {
    { cont.data() } -> std::convertible_to<T*>;
    { cont.size() } -> std::convertible_to<size_t>;
  }
{
  return span<T>{cont}.subspan(offset, size);
}
} // namespace sxt::basct
