#pragma once

#include "sxt/field51/type/element.h"

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// element_cached
//--------------------------------------------------------------------------------------------------
struct element_cached {
  f51t::element YplusX;
  f51t::element YminusX;
  f51t::element Z;
  f51t::element T2d;
};
} // namespace sxt::c21t
