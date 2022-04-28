#pragma once

#include "sxt/field51/type/element.h"

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// element_p1p1
//--------------------------------------------------------------------------------------------------
/**
 *   (completed): ((X:Z),(Y:T)) satisfying x=X/Z, y=Y/T
 */
struct element_p1p1 {
  f51t::element X;
  f51t::element Y;
  f51t::element Z;
  f51t::element T;
};
} // namespace sxt::c21t
