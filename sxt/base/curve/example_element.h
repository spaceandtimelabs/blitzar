#pragma once

#include <cstdint>
#include <iostream>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::bascrv {
//--------------------------------------------------------------------------------------------------
// element97
//--------------------------------------------------------------------------------------------------
struct element97 {
  uint32_t value;
  bool marked = false;

  element97() noexcept = default;

  constexpr element97(uint32_t val) noexcept : value{val % 97u} {}

  static constexpr element97 identity() noexcept {
    return {0};
  }

  bool operator==(const element97&) const noexcept = default;
};

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& out, const element97& e) noexcept {
  out << e.value;
  return out;
}

//--------------------------------------------------------------------------------------------------
// double_element
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void double_element(element97& res, const element97& e) noexcept {
  res.value = (e.value + e.value) % 97u;
}

//--------------------------------------------------------------------------------------------------
// neg
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void neg(element97& res, const element97& e) noexcept {
  res.value = (97u - e.value) % 97u;
}

//--------------------------------------------------------------------------------------------------
// cneg
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void cneg(element97& res, int b) noexcept {
  if (b) {
    res.value = (97u - res.value) % 97u;
  }
}

//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void add(element97& res, const element97& x, const element97& y) noexcept {
  res.value = (x.value + y.value) % 97u;
}

//--------------------------------------------------------------------------------------------------
// add_inplace
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void add_inplace(element97& res, element97& x) noexcept {
  res.value = (res.value + x.value) % 97u;
  x = 13; // simulate a destructive add by setting x to an arbitrary value 
}

//--------------------------------------------------------------------------------------------------
// mark
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void mark(element97& res) noexcept {
  res.marked = true;
}

//--------------------------------------------------------------------------------------------------
// is_marked
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE bool is_marked(const element97& e) noexcept {
  return e.marked;
}
} // namespace sxt::bascrv
