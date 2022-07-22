#pragma once

#include <array>

#include "sxt/base/num/hex.h"
#include "sxt/field51/base/byte_conversion.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51t {
//--------------------------------------------------------------------------------------------------
// parse_literal_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
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
} // namespace detail

//--------------------------------------------------------------------------------------------------
// parse_literal
//--------------------------------------------------------------------------------------------------
namespace detail {
template <char Z, char X, char... Digits>
void parse_literal(std::array<uint64_t, 4>& bytes) noexcept {
  static_assert(Z == '0' && X == 'x' && sizeof...(Digits) > 0 && sizeof...(Digits) <= 64,
                "invalid field literal");
  parse_literal_impl<Digits...>(bytes);
}
} // namespace detail

//--------------------------------------------------------------------------------------------------
// _f51
//--------------------------------------------------------------------------------------------------
template <char... Chars> element operator"" _f51() noexcept {
  std::array<uint64_t, 4> bytes = {};
  detail::parse_literal<Chars...>(bytes);
  element res;
  f51b::from_bytes(res.data(), reinterpret_cast<const uint8_t*>(bytes.data()));
  return res;
}
} // namespace sxt::f51t
