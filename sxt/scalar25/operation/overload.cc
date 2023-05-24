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
#include "sxt/scalar25/operation/overload.h"

#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/operation/inv.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/neg.h"
#include "sxt/scalar25/operation/sub.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25t {
//--------------------------------------------------------------------------------------------------
// operator+
//--------------------------------------------------------------------------------------------------
element operator+(const element& lhs, const element& rhs) noexcept {
  element res;
  s25o::add(res, lhs, rhs);
  return res;
}

//--------------------------------------------------------------------------------------------------
// operator-
//--------------------------------------------------------------------------------------------------
element operator-(const element& lhs, const element& rhs) noexcept {
  element res;
  s25o::sub(res, lhs, rhs);
  return res;
}

//--------------------------------------------------------------------------------------------------
// operator*
//--------------------------------------------------------------------------------------------------
element operator*(const element& lhs, const element& rhs) noexcept {
  element res;
  s25o::mul(res, lhs, rhs);
  return res;
}

//--------------------------------------------------------------------------------------------------
// operator/
//--------------------------------------------------------------------------------------------------
element operator/(const element& lhs, const element& rhs) noexcept {
  element t;
  s25o::inv(t, rhs);
  return lhs * t;
}

//--------------------------------------------------------------------------------------------------
// operator-
//--------------------------------------------------------------------------------------------------
element operator-(const element& x) noexcept {
  element res;
  s25o::neg(res, x);
  return res;
}

//--------------------------------------------------------------------------------------------------
// operator+=
//--------------------------------------------------------------------------------------------------
element& operator+=(element& lhs, const element& rhs) noexcept {
  lhs = lhs + rhs;
  return lhs;
}

//--------------------------------------------------------------------------------------------------
// operator-=
//--------------------------------------------------------------------------------------------------
element& operator-=(element& lhs, const element& rhs) noexcept {
  lhs = lhs - rhs;
  return lhs;
}

//--------------------------------------------------------------------------------------------------
// operator*=
//--------------------------------------------------------------------------------------------------
element& operator*=(element& lhs, const element& rhs) noexcept {
  lhs = lhs * rhs;
  return lhs;
}

//--------------------------------------------------------------------------------------------------
// operator/=
//--------------------------------------------------------------------------------------------------
element& operator/=(element& lhs, const element& rhs) noexcept {
  lhs = lhs / rhs;
  return lhs;
}
} // namespace sxt::s25t
