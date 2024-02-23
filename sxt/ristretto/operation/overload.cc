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
#include "sxt/ristretto/operation/overload.h"

#include "sxt/curve32/operation/overload.h"
#include "sxt/curve32/type/element_p3.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::s25t {
class element;
}

namespace sxt::rstt {
//--------------------------------------------------------------------------------------------------
// operator+
//--------------------------------------------------------------------------------------------------
compressed_element operator+(const compressed_element& lhs,
                             const compressed_element& rhs) noexcept {
  c32t::element_p3 lhs_p, rhs_p;
  rsto::decompress(lhs_p, lhs);
  rsto::decompress(rhs_p, rhs);
  auto res_p = lhs_p + rhs_p;
  compressed_element res;
  rsto::compress(res, res_p);
  return res;
}

compressed_element operator+(const c32t::element_p3& lhs, const compressed_element& rhs) noexcept {
  compressed_element lhs_p;
  rsto::compress(lhs_p, lhs);
  return lhs_p + rhs;
}

compressed_element operator+(const compressed_element& lhs, const c32t::element_p3& rhs) noexcept {
  return rhs + lhs;
}

//--------------------------------------------------------------------------------------------------
// operator-
//--------------------------------------------------------------------------------------------------
compressed_element operator-(const compressed_element& lhs,
                             const compressed_element& rhs) noexcept {
  c32t::element_p3 lhs_p, rhs_p;
  rsto::decompress(lhs_p, lhs);
  rsto::decompress(rhs_p, rhs);
  auto res_p = lhs_p - rhs_p;
  compressed_element res;
  rsto::compress(res, res_p);
  return res;
}

compressed_element operator-(const compressed_element& x) noexcept {
  c32t::element_p3 x_p;
  rsto::decompress(x_p, x);
  auto res_p = -x_p;
  compressed_element res;
  rsto::compress(res, res_p);
  return res;
}

compressed_element operator-(const c32t::element_p3& lhs, const compressed_element& rhs) noexcept {
  compressed_element lhs_p;
  rsto::compress(lhs_p, lhs);
  return lhs_p - rhs;
}

compressed_element operator-(const compressed_element& lhs, const c32t::element_p3& rhs) noexcept {
  compressed_element rhs_p;
  rsto::compress(rhs_p, rhs);
  return lhs - rhs_p;
}

//--------------------------------------------------------------------------------------------------
// operator*
//--------------------------------------------------------------------------------------------------
compressed_element operator*(const s25t::element& lhs, const compressed_element& rhs) noexcept {
  c32t::element_p3 rhs_p;
  rsto::decompress(rhs_p, rhs);
  auto res_p = lhs * rhs_p;
  compressed_element res;
  rsto::compress(res, res_p);
  return res;
}

//--------------------------------------------------------------------------------------------------
// operator+=
//--------------------------------------------------------------------------------------------------
compressed_element& operator+=(compressed_element& lhs, const compressed_element& rhs) noexcept {
  c32t::element_p3 lhs_p, rhs_p;
  rsto::decompress(lhs_p, lhs);
  rsto::decompress(rhs_p, rhs);
  lhs_p += rhs_p;
  rsto::compress(lhs, lhs_p);
  return lhs;
}

//--------------------------------------------------------------------------------------------------
// operator-=
//--------------------------------------------------------------------------------------------------
compressed_element& operator-=(compressed_element& lhs, const compressed_element& rhs) noexcept {
  c32t::element_p3 lhs_p, rhs_p;
  rsto::decompress(lhs_p, lhs);
  rsto::decompress(rhs_p, rhs);
  lhs_p -= rhs_p;
  rsto::compress(lhs, lhs_p);
  return lhs;
}
} // namespace sxt::rstt
