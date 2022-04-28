#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/neg.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// cneg
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void cneg(f51t::element& h, const f51t::element& f,
                 unsigned int b) noexcept {
  f51t::element negf;

  f51o::neg(negf, f);
  h = f;
  f51o::cmov(h, negf, b);
}
}  // namespace sxt::f51o
