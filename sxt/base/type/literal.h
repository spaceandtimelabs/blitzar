#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "sxt/base/num/hex.h"

namespace sxt::bast {
//--------------------------------------------------------------------------------------------------
// parse_literal_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <size_t N, char First, char... Rest>
void parse_literal_impl(std::array<uint64_t, N>& bytes) noexcept {
  static_assert(basn::is_hex_digit(First), "invalid digit");
  auto digit_index = sizeof...(Rest);
  auto element_index = digit_index / 16;
  auto offset = (digit_index % 16) * 4;
  uint64_t val = basn::to_hex_value(First);
  bytes[element_index] += val << offset;
  if constexpr (sizeof...(Rest) > 0) {
    parse_literal_impl<N, Rest...>(bytes);
  }
}
} // namespace detail

//--------------------------------------------------------------------------------------------------
// parse_literal
//--------------------------------------------------------------------------------------------------
template <size_t N, char Z, char X, char... Digits>
void parse_literal(std::array<uint64_t, N>& bytes) noexcept {
  static_assert(Z == '0' && X == 'x' && sizeof...(Digits) > 0 && sizeof...(Digits) <= N * 16,
                "invalid field literal");
  bytes = {};
  detail::parse_literal_impl<N, Digits...>(bytes);
}
} // namespace sxt::bast
