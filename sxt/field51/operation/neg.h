#pragma once

#include "sxt/base/macro/cuda_callable.h"

#include "sxt/field51/constant/zero.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// neg
//--------------------------------------------------------------------------------------------------
/*
 h = -f
 */
CUDA_CALLABLE
inline void neg(f51t::element& h, const f51t::element& f) noexcept {
  sub(h, f51t::element{f51cn::zero_v}, f);
}
} // namespace sxt::f51o
