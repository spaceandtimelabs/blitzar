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
#include <cstring>

#include "sxt/base/type/literal.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25t {
//--------------------------------------------------------------------------------------------------
// _s25
//--------------------------------------------------------------------------------------------------
template <char... Chars> element operator"" _s25() noexcept {
  std::array<uint64_t, 4> bytes = {};
  bast::parse_literal<4, Chars...>(bytes);
  element res;
  std::memcpy(static_cast<void*>(res.data()), static_cast<const void*>(bytes.data()), 32);
  return res;
}
} // namespace sxt::s25t
