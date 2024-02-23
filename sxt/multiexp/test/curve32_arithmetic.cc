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
#include "sxt/multiexp/test/curve32_arithmetic.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "sxt/base/container/stack_array.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/abs.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/constexpr_switch.h"
#include "sxt/base/type/int.h"
#include "sxt/curve32/operation/add.h"
#include "sxt/curve32/operation/neg.h"
#include "sxt/curve32/operation/scalar_multiply.h"
#include "sxt/curve32/property/curve.h"
#include "sxt/curve32/type/element_p3.h"
#include "sxt/multiexp/base/exponent_sequence.h"

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// read_exponent
//--------------------------------------------------------------------------------------------------
static void read_exponent(c32t::element_p3& e, basct::span<uint8_t>& exponent,
                          const mtxb::exponent_sequence& sequence, size_t index) noexcept {
  uint8_t element_nbytes = sequence.element_nbytes;
  exponent = {exponent.data(), element_nbytes};
  std::copy_n(sequence.data + index * element_nbytes, element_nbytes, exponent.data());
  if (!static_cast<bool>(sequence.is_signed)) {
    return;
  }
  basn::constexpr_switch<5>(
      basn::ceil_log2(element_nbytes),
      [&]<unsigned NumBytesLg2>(std::integral_constant<unsigned, NumBytesLg2>) noexcept {
        static constexpr auto NumBytes = 1ull << NumBytesLg2;
        bast::sized_int_t<NumBytes * 8> x{};
        std::copy_n(exponent.begin(), element_nbytes, reinterpret_cast<uint8_t*>(&x));
        auto abs_x = basn::abs(x);
        if (x == abs_x) {
          return;
        }
        c32o::neg(e, e);
        std::copy_n(reinterpret_cast<uint8_t*>(&abs_x), element_nbytes, exponent.begin());
      });
}

//--------------------------------------------------------------------------------------------------
// sum_curve32_elements
//--------------------------------------------------------------------------------------------------
void sum_curve32_elements(basct::span<c32t::element_p3> result,
                          basct::cspan<basct::cspan<uint64_t>> terms,
                          basct::cspan<c32t::element_p3> inputs) noexcept {
  SXT_RELEASE_ASSERT(result.size() == terms.size());
  for (size_t result_index = 0; result_index < result.size(); ++result_index) {
    auto& res_i = result[result_index];
    res_i = c32t::element_p3::identity();
    for (auto term_index : terms[result_index]) {
      SXT_RELEASE_ASSERT(term_index < inputs.size());
      c32o::add(res_i, res_i, inputs[term_index]);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// mul_sum_curve32_elements
//--------------------------------------------------------------------------------------------------
void mul_sum_curve32_elements(basct::span<c32t::element_p3> result,
                              basct::cspan<c32t::element_p3> generators,
                              basct::cspan<mtxb::exponent_sequence> sequences) noexcept {
  SXT_RELEASE_ASSERT(result.size() == sequences.size());
  SXT_STACK_ARRAY(exponent, 32, uint8_t);
  for (size_t output_index = 0; output_index < result.size(); ++output_index) {
    c32t::element_p3 output = c32t::element_p3::identity();
    auto sequence = sequences[output_index];
    SXT_RELEASE_ASSERT(sequence.n <= generators.size());
    for (size_t generator_index = 0; generator_index < sequence.n; ++generator_index) {
      auto e = generators[generator_index];
      read_exponent(e, exponent, sequence, generator_index);
      if (exponent.size() == 32) {
        // split multiplication into multiple steps to avoid a scalar25 reduction -- this allows us
        // to compare curve32 elements directly
        c32o::scalar_multiply(e, exponent.subspan(1), e);
        c32o::scalar_multiply(e, 1ul << 8, e);
        c32o::add(output, output, e);
        e = generators[generator_index];
        c32o::scalar_multiply(e, exponent.subspan(0, 1), e);
      } else {
        c32o::scalar_multiply(e, exponent, e);
      }
      SXT_RELEASE_ASSERT(c32p::is_on_curve(e));
      c32o::add(output, output, e);
    }
    result[output_index] = output;
  }
}
} // namespace sxt::mtxtst
