/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/multiexp/curve21/multiproducts_combination.h"

#include <algorithm>
#include <tuple>

#include "sxt/base/bit/span_op.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/base/container/stack_array.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/curve21/doubling_reduction.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// init_output_products
//--------------------------------------------------------------------------------------------------
static std::tuple<basct::cspan<c21t::element_p3>, basct::cspan<uint8_t>>
init_output_products(size_t& product_index, size_t& input_index,
                     basct::span<c21t::element_p3> products, basct::blob_array& output_digit_or_all,
                     bool is_signed) noexcept {
  auto digit_or_all = output_digit_or_all[product_index++];
  size_t digit_count_one = basbt::pop_count(digit_or_all);
  if (!is_signed) {
    auto output_products = products.subspan(input_index, digit_count_one);
    input_index += digit_count_one;
    return {output_products, digit_or_all};
  }
  SXT_STACK_ARRAY(pos_digit_or_all, digit_or_all.size(), uint8_t);
  std::copy(digit_or_all.begin(), digit_or_all.end(), pos_digit_or_all.begin());

  auto neg_digit_or_all = output_digit_or_all[product_index++];
  basbt::or_equal(digit_or_all, neg_digit_or_all);

  auto pos_digit_count_one = digit_count_one;
  auto neg_digit_count_one = basbt::pop_count(neg_digit_or_all);
  digit_count_one = basbt::pop_count(digit_or_all);

  auto output_products = products.subspan(input_index, digit_count_one);
  SXT_STACK_ARRAY(pos_products, pos_digit_count_one, c21t::element_p3);
  std::copy_n(output_products.begin(), pos_digit_count_one, pos_products.begin());
  auto neg_products = products.subspan(input_index + pos_digit_count_one, neg_digit_count_one);
  for (auto& e : neg_products) {
    c21o::neg(e, e);
  }

  size_t pos_index = 0;
  size_t neg_index = 0;
  size_t out_index = 0;
  basbt::for_each_bit(digit_or_all, [&](size_t bit_index) noexcept {
    auto is_pos_set = basbt::test_bit(pos_digit_or_all, bit_index);
    auto is_neg_set = basbt::test_bit(neg_digit_or_all, bit_index);
    if (is_pos_set && is_neg_set) {
      c21o::add(output_products[out_index++], pos_products[pos_index++], neg_products[neg_index++]);
    } else if (is_pos_set) {
      output_products[out_index++] = pos_products[pos_index++];
    } else {
      output_products[out_index++] = neg_products[neg_index++];
    }
  });

  input_index += pos_digit_count_one + neg_digit_count_one;
  return {output_products, digit_or_all};
}

//--------------------------------------------------------------------------------------------------
// combine_multiproducts
//--------------------------------------------------------------------------------------------------
void combine_multiproducts(basct::span<c21t::element_p3> outputs,
                           const basct::blob_array& output_digit_or_all,
                           basct::cspan<c21t::element_p3> products) noexcept {
  size_t input_index = 0;
  for (size_t output_index = 0; output_index < output_digit_or_all.size(); ++output_index) {
    auto digit_or_all = output_digit_or_all[output_index];
    int digit_count_one = basbt::pop_count(digit_or_all);
    if (digit_count_one == 0) {
      outputs[output_index] = c21cn::zero_p3_v;
      continue;
    }
    c21t::element_p3 output;
    SXT_DEBUG_ASSERT(input_index + digit_count_one <= products.size());
    doubling_reduce(output, digit_or_all,
                    basct::cspan<c21t::element_p3>{&products[input_index],
                                                   static_cast<size_t>(digit_count_one)});
    input_index += digit_count_one;
    outputs[output_index] = output;
  }
}

void combine_multiproducts(basct::span<c21t::element_p3> outputs,
                           basct::blob_array& output_digit_or_all,
                           basct::span<c21t::element_p3> products,
                           basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  auto num_sequences = exponents.size();
  // clang-format off
  SXT_DEBUG_ASSERT(
      outputs.size() == num_sequences && 
      exponents.size() == num_sequences
  );
  // clang-format on
  size_t product_index = 0;
  size_t input_index = 0;
  for (size_t sequence_index = 0; sequence_index < exponents.size(); ++sequence_index) {
    auto sequence = exponents[sequence_index];
    auto [output_products, digit_or_all] =
        init_output_products(product_index, input_index, products, output_digit_or_all,
                             static_cast<bool>(sequence.is_signed));
    if (products.empty()) {
      outputs[sequence_index] = c21cn::zero_p3_v;
      continue;
    }
    doubling_reduce(outputs[sequence_index], digit_or_all, output_products);
  }
}
} // namespace sxt::mtxc21
