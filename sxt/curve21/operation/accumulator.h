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

  CUDA_CALLABLE static void accumulate_inplace(volatile c21t::element_p3& res,
                                               volatile c21t::element_p3& e) noexcept {
    add_inplace(res, e);
  }

  CUDA_CALLABLE static void accumulate_inplace(c21t::element_p3& res,
                                               c21t::element_p3& e) noexcept {
    add_inplace(res, e);
  }
};
} // namespace sxt::c21o
