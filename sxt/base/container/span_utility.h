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
