#pragma once

#include <array>
#include <cstdint>

#include "sxt/base/num/hex.h"

namespace sxt::bast {
//--------------------------------------------------------------------------------------------------
// parse_literal_impl
//--------------------------------------------------------------------------------------------------
template <char First, char... Rest>
void parse_literal_impl(std::array<uint64_t, 4>& bytes) noexcept {
  static_assert(basn::is_hex_digit(First), "invalid digit");
  auto digit_index = sizeof...(Rest);
  auto element_index = digit_index / 16;
  auto offset = (digit_index % 16) * 4;
  uint64_t val = basn::to_hex_value(First);
  bytes[element_index] += val << offset;
  if constexpr (sizeof...(Rest) > 0) {
    parse_literal_impl<Rest...>(bytes);
  }
}

//--------------------------------------------------------------------------------------------------
// parse_literal
//--------------------------------------------------------------------------------------------------
template <char Z, char X, char... Digits>
void parse_literal(std::array<uint64_t, 4>& bytes) noexcept {
  static_assert(Z == '0' && X == 'x' && sizeof...(Digits) > 0 && sizeof...(Digits) <= 64,
                "invalid field literal");
  parse_literal_impl<Digits...>(bytes);
}
} // namespace sxt::bast
