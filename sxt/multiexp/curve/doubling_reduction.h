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

#include <concepts>

#include "sxt/base/bit/count.h"
#include "sxt/base/bit/span_op.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/error/assert.h"

namespace sxt::mtxcrv {
//--------------------------------------------------------------------------------------------------
// doubling_reduce
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element, class Inputs>
  requires std::constructible_from<basct::cspan<Element>, Inputs>
void doubling_reduce(Element& res, basct::cspan<uint8_t> digit_or_all,
                     const Inputs& inputs_convertible) noexcept {
  basct::cspan<Element> inputs{inputs_convertible};
  SXT_DEBUG_ASSERT(!inputs.empty() && inputs.size() == basbt::pop_count(digit_or_all));

  auto input_index = inputs.size();
  size_t digit_bit_index = 8 * digit_or_all.size() - basbt::count_leading_zeros(digit_or_all) - 1;

  // we manually set the first `input_index`
  // to prevent one `double` and one `add` operation
  res = inputs[--input_index];

  // The following implementation uses the formula:
  // output = 2^{i_0} * a0 + 2^{i_1} * (a1 + 2^{i_2} * (a2 + ..)..),
  // where i_0 <= i_1 <= i_2 <= ... <= i_n.
  while (digit_bit_index-- > 0) {
    double_element(res, res);
    if (basbt::test_bit(digit_or_all, digit_bit_index)) {
      add(res, res, inputs[--input_index]);
    }
  }
}
} // namespace sxt::mtxcrv
