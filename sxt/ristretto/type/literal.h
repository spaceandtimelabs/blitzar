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
#pragma once

#include "sxt/base/type/literal.h"
#include "sxt/curve32/type/element_p3.h"
#include "sxt/field32/base/byte_conversion.h"
#include "sxt/field32/type/element.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/base/point_formation.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::rstt {
//--------------------------------------------------------------------------------------------------
// _rs
//--------------------------------------------------------------------------------------------------
template <char... Chars> c32t::element_p3 operator"" _rs() noexcept {
  std::array<uint64_t, 8> bytes;
  bast::parse_literal<8, Chars...>(bytes);
  f32t::element x, y;
  f32b::from_bytes(x.data(), reinterpret_cast<const uint8_t*>(bytes.data()));
  f32b::from_bytes(y.data(), reinterpret_cast<const uint8_t*>(&bytes[4]));
  c32t::element_p3 res;
  rstb::form_ristretto_point(res, x, y);
  return res;
}

//--------------------------------------------------------------------------------------------------
// _crs
//--------------------------------------------------------------------------------------------------
template <char... Chars> compressed_element operator"" _crs() noexcept {
  auto e = operator""_rs < Chars... > ();
  compressed_element res;
  rstb::to_bytes(res.data(), e);
  return res;
}
} // namespace sxt::rstt
