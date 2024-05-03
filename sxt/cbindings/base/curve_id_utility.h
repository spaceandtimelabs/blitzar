#pragma once

#include "sxt/base/error/panic.h"
#include "sxt/base/type/type.h"
#include "sxt/cbindings/base/curve_id.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve_bng1/operation/add.h"
#include "sxt/curve_bng1/operation/double.h"
#include "sxt/curve_bng1/operation/neg.h"
#include "sxt/curve_bng1/type/element_p2.h"
#include "sxt/curve_g1/operation/add.h"
#include "sxt/curve_g1/operation/double.h"
#include "sxt/curve_g1/operation/neg.h"
#include "sxt/curve_g1/type/element_p2.h"

namespace sxt::cbnb {
//--------------------------------------------------------------------------------------------------
// switch_curve_type
//--------------------------------------------------------------------------------------------------
template <class F> void switch_curve_type(curve_id_t id, F f) {
  switch(id) {
    case curve_id_t::curve21:
      f(bast::type_t<c21t::element_p3>{});
    case curve_id_t::bls381:
      f(bast::type_t<cg1t::element_p2>{});
    case curve_id_t::bnp:
      f(bast::type_t<cn1t::element_p2>{});
    default:
      baser::panic("unsupported curve id {}", static_cast<unsigned>(id));
  }
}
} // namespace sxt::cbnb
