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
#include "sxt/multiexp/curve21/doubling_reduction.h"

#include "sxt/base/bit/count.h"
#include "sxt/base/bit/span_op.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// doubling_reduce
//--------------------------------------------------------------------------------------------------
void doubling_reduce(c21t::element_p3& res, basct::cspan<uint8_t> digit_or_all,
                     basct::cspan<c21t::element_p3> inputs) noexcept {
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
    c21o::double_element(res, res);
    if (basbt::test_bit(digit_or_all, digit_bit_index)) {
      c21o::add(res, res, inputs[--input_index]);
    }
  }
}
} // namespace sxt::mtxc21
