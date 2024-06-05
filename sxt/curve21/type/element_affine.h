#pragma once

#include "sxt/field51/type/element.h"

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// element_affine
//--------------------------------------------------------------------------------------------------
struct element_affine {
  f51t::element X;
  f51t::element Y;

  static constexpr element_affine identity() noexcept {
    return element_affine{f51t::element{0, 0, 0, 0, static_cast<uint64_t>(-1)},
                          f51t::element{0, 0, 0, 0, 0}};
  }
};

//--------------------------------------------------------------------------------------------------
// is_identity 
//--------------------------------------------------------------------------------------------------
inline bool is_identity(const element_affine& e) noexcept {
  return e.X[4] == static_cast<uint64_t>(-1);
}
} // namespace sxt::c21t
