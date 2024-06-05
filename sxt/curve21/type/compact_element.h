#pragma once

#include "sxt/field51/type/element.h"
#include "sxt/field51/constant/one.h"
#include "sxt/field51/constant/zero.h"

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// compact_element
//--------------------------------------------------------------------------------------------------
struct compact_element {
  f51t::element X;
  f51t::element Y;
  f51t::element T;

  static constexpr compact_element identity() noexcept {
    return compact_element{f51cn::zero_v, f51cn::one_v, f51cn::one_v};
  }
};
} // namespace sxt::c21t
