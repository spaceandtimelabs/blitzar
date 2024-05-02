#pragma once

#include "sxt/cbindings/base/curve_id.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/curve_bng1/type/element_p2.h"

namespace sxt::cbnb {
//--------------------------------------------------------------------------------------------------
// switch_curve_type
//--------------------------------------------------------------------------------------------------
template <class F> void switch_curve_type(curve_id_t curve, F f) {
  (void)curve;
  (void)f;
}
} // namespace sxt::cbnb
