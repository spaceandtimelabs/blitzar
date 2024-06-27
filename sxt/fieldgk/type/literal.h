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

#include <array>

#include "sxt/base/type/literal.h"
#include "sxt/fieldgk/base/byte_conversion.h"
#include "sxt/fieldgk/type/element.h"

namespace sxt::fgkt {
//--------------------------------------------------------------------------------------------------
// _fgk
//--------------------------------------------------------------------------------------------------
template <char... Chars> element operator"" _fgk() noexcept {
  std::array<uint64_t, 4> bytes = {};
  bast::parse_literal<4, Chars...>(bytes);
  element res;
  bool is_below;
  fgkb::from_bytes_le(is_below, res.data(), reinterpret_cast<const uint8_t*>(bytes.data()));
  return res;
}
} // namespace sxt::fgkt
