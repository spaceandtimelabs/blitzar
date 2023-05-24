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

#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// multiproduct_bitset_operator
//--------------------------------------------------------------------------------------------------
class multiproduct_bitset_operator {
  static constexpr uint64_t unset_marker_v = static_cast<uint64_t>(-1);

public:
  void mark_unset(c21t::element_p3& e) const noexcept { e.Z[4] = unset_marker_v; }

  bool is_set(const c21t::element_p3& e) const noexcept { return e.Z[4] != unset_marker_v; }

  void add(c21t::element_p3& res, const c21t::element_p3& lhs,
           const c21t::element_p3& rhs) const noexcept {
    c21o::add(res, lhs, rhs);
  }
};
} // namespace sxt::mtxc21
