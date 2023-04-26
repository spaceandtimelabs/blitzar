#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/curve21/type/conversion_utility.h"
#include "sxt/curve21/type/double_impl.h"
#include "sxt/curve21/type/element_p1p1.h"
#include "sxt/curve21/type/element_p2.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// double_element
//--------------------------------------------------------------------------------------------------
/*
 r = 2 * p
*/
CUDA_CALLABLE
inline void double_element(c21t::element_p1p1& r, const c21t::element_p2& p) noexcept {
  c21t::double_element_impl(r, p);
}

CUDA_CALLABLE
inline void double_element(c21t::element_p1p1& r, const c21t::element_p3& p) noexcept {
  c21t::double_element_impl(r, p);
}

CUDA_CALLABLE
inline void double_element(c21t::element_p3& r, const c21t::element_p3& p) noexcept {
  c21t::element_p1p1 t;
  double_element(t, p);
  to_element_p3(r, t);
}
} // namespace sxt::c21o
