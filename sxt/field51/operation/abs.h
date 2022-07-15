#pragma once

#include "sxt/base/macro/cuda_callable.h"

#include "sxt/field51/operation/cneg.h"
#include "sxt/field51/property/sign.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// abs
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void abs(f51t::element& h, const f51t::element& f) noexcept {
  f51o::cneg(h, f, f51p::is_negative(f));
}
} // namespace sxt::f51o
