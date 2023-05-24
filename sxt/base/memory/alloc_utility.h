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
#include <type_traits>

#include "sxt/base/memory/alloc.h"

namespace sxt::basm {
//--------------------------------------------------------------------------------------------------
// allocate_array
//--------------------------------------------------------------------------------------------------
template <class T, std::enable_if_t<std::is_trivially_destructible_v<T>>* = nullptr>
T* allocate_array(alloc_t alloc, size_t n) noexcept {
  // Use a form that will work with c++17. In C++20, we'd use
  // allocate_bytes
  return static_cast<T*>(alloc.resource()->allocate(n * sizeof(T), alignof(T)));
}

//--------------------------------------------------------------------------------------------------
// allocate_object
//--------------------------------------------------------------------------------------------------
template <class T, std::enable_if_t<std::is_trivially_destructible_v<T>>* = nullptr>
T* allocate_object(alloc_t alloc) noexcept {
  return allocate_array<T>(alloc, 1);
}
} // namespace sxt::basm
