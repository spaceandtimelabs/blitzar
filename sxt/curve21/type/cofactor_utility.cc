/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/curve21/type/cofactor_utility.h"

#include "sxt/curve21/type/conversion_utility.h"
#include "sxt/curve21/type/double_impl.h"
#include "sxt/curve21/type/element_p1p1.h"
#include "sxt/curve21/type/element_p2.h"

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// clear_cofactor
//--------------------------------------------------------------------------------------------------
void clear_cofactor(element_p3& p3) noexcept {
  element_p1p1 p1;
  element_p2 p2;

  double_element_impl(p1, p3);
  to_element_p2(p2, p1);
  double_element_impl(p1, p2);
  to_element_p2(p2, p1);
  double_element_impl(p1, p2);
  to_element_p3(p3, p1);
}
} // namespace sxt::c21t
