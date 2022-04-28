#pragma once

#include "sxt/field51/type/element.h"

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// element_p2
//--------------------------------------------------------------------------------------------------
/**
 * (projective): (X:Y:Z) satisfying x=X/Z, y=Y/Z
 */
struct element_p2 {
  f51t::element X;
  f51t::element Y;
  f51t::element Z;
};
} // namespace sxt::c21t
