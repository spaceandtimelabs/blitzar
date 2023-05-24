/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2023
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
