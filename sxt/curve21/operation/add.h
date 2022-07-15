#pragma once

#include "sxt/base/macro/cuda_callable.h"

#include "sxt/curve21/type/conversion_utility.h"
#include "sxt/curve21/type/element_cached.h"
#include "sxt/curve21/type/element_p1p1.h"

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
} // namespace sxt::c21o
