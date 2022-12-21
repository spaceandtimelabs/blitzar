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
