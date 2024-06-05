#pragma once

#include "sxt/field51/type/element.h"
#include "sxt/field51/constant/one.h"
#include "sxt/field51/constant/zero.h"

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// compact_element
//--------------------------------------------------------------------------------------------------
struct element_affine {
  f51t::element X;
  f51t::element Y;
  f51t::element T;

  static constexpr element_affine identity() noexcept {
    return element_affine{f51cn::zero_v, f51cn::one_v, f51cn::one_v};
  }
};
} // namespace sxt::c21t
