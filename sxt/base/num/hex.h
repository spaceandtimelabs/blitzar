#pragma once

#include <cstdint>

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// is_hex_digit
//--------------------------------------------------------------------------------------------------
constexpr bool is_hex_digit(char c) noexcept {
  if ('0' <= c && c <= '9') {
    return true;
  }
  if ('a' <= c && c <= 'f') {
    return true;
  }
  if ('A' <= c && c <= 'F') {
    return true;
  }
  return false;
}

//--------------------------------------------------------------------------------------------------
// to_hex_value
//--------------------------------------------------------------------------------------------------
constexpr uint8_t to_hex_value(char c) noexcept {
  if ('0' <= c && c <= '9') {
    return static_cast<uint8_t>(c - '0');
  }
  if ('a' <= c && c <= 'f') {
    return static_cast<uint8_t>(c - 'a' + 10);
  }
  if ('A' <= c && c <= 'F') {
    return static_cast<uint8_t>(c - 'A' + 10);
  }
  return static_cast<uint8_t>(-1);
}
} // namesape sxt::basn
