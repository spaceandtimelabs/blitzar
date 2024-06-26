/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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

#include "sxt/fieldgk/constant/one.h"
#include "sxt/fieldgk/type/element.h"

namespace sxt::cgkt {
//--------------------------------------------------------------------------------------------------
// compact_element
//--------------------------------------------------------------------------------------------------
struct compact_element {
  fgkt::element X;
  fgkt::element Y;

  constexpr bool is_identity() const noexcept { return X[3] == static_cast<uint64_t>(-1); }

  static constexpr compact_element identity() noexcept {
    return {
        {0, 0, 0, static_cast<uint64_t>(-1)},
        fgkcn::one_v,
    };
  }
};
} // namespace sxt::cgkt
