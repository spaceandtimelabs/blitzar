#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// accumulator
//--------------------------------------------------------------------------------------------------
struct accumulator {
  using value_type = c21t::element_p3;

  CUDA_CALLABLE static void accumulate(c21t::element_p3& res, const c21t::element_p3& e) noexcept {
    c21o::add(res, res, e);
  }

  CUDA_CALLABLE static void accumulate(volatile c21t::element_p3& res,
                                       const volatile c21t::element_p3& e) noexcept {
    add(res, res, e);
  }
};
} // namespace sxt::c21o
