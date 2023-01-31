#pragma once

#include <cstring>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/operation/product_mapper.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// accumulator
//--------------------------------------------------------------------------------------------------
struct accumulator {
  using value_type = s25t::element;

  CUDA_CALLABLE static void accumulate(s25t::element& res, product_mapper mapper,
                                       unsigned int index) noexcept {
    s25o::muladd(res, mapper.lhs_data()[index], mapper.rhs_data()[index], res);
  }

  CUDA_CALLABLE static void accumulate(s25t::element& res, const s25t::element& e) noexcept {
    s25o::add(res, res, e);
  }

  CUDA_CALLABLE static void accumulate(volatile s25t::element& res,
                                       const volatile s25t::element& e) noexcept {
    s25t::element res_p;
    s25o::add(res_p, res, e);
    res = res_p;
  }
};
} // namespace sxt::s25o
