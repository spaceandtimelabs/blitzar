#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/curve21/type/conversion_utility.h"
#include "sxt/curve21/type/element_cached.h"
#include "sxt/curve21/type/element_p1p1.h"
#include "sxt/curve21/type/element_p3.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/*
 r = p + q
 */
CUDA_CALLABLE
void add(c21t::element_p1p1& r, const c21t::element_p3& p, const c21t::element_cached& q) noexcept;

CUDA_CALLABLE
inline void add(c21t::element_p3& res, const c21t::element_p3& lhs,
                const c21t::element_p3& rhs) noexcept {
  c21t::element_cached t;
  to_element_cached(t, rhs);
  c21t::element_p1p1 res_p;
  add(res_p, lhs, t);
  to_element_p3(res, res_p);
}

CUDA_CALLABLE
inline void add(volatile c21t::element_p3& res, const volatile c21t::element_p3& lhs,
                const volatile c21t::element_p3& rhs) noexcept {
  c21t::element_p3 x{
      .X{rhs.X},
      .Y{rhs.Y},
      .Z{rhs.Z},
      .T{rhs.T},
  };
  c21t::element_cached t;
  to_element_cached(t, x);
  c21t::element_p1p1 res_p;
  x = {
      .X{lhs.X},
      .Y{lhs.Y},
      .Z{lhs.Z},
      .T{lhs.T},
  };
  add(res_p, x, t);
  to_element_p3(res, res_p);
}
} // namespace sxt::c21o
