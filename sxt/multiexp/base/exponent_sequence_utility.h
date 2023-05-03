#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include "sxt/multiexp/base/exponent_sequence.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// to_exponent_sequence
//--------------------------------------------------------------------------------------------------
template <class Cont, class T = std::remove_cvref_t<decltype(*std::declval<const Cont>().data())>>
exponent_sequence to_exponent_sequence(const Cont& cont) noexcept
  requires requires {
    { cont.data() } -> std::convertible_to<const T*>;
    { cont.size() } -> std::convertible_to<uint64_t>;
  }
{
  static_assert(sizeof(T) <= 32, "element is too large");
  const T* data = cont.data();
  int is_signed = 0;
  if constexpr (std::is_signed_v<T>) {
    is_signed = 1;
  }
  return exponent_sequence{
      .element_nbytes = sizeof(T),
      .n = cont.size(),
      .data = reinterpret_cast<const uint8_t*>(data),
      .is_signed = is_signed,
  };
}
} // namespace sxt::mtxb
