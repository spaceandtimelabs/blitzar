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

#include <algorithm>
#include <concepts>
#include <tuple>

#include "sxt/base/bit/span_op.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/base/container/span.h"
#include "sxt/base/container/stack_array.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/error/assert.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/management/managed_array_fwd.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/curve/doubling_reduction.h"

namespace sxt::basct {
class blob_array;
}

namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxcrv {
//--------------------------------------------------------------------------------------------------
// init_output_products
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
static std::tuple<basct::cspan<Element>, basct::cspan<uint8_t>>
init_output_products(size_t& product_index, size_t& input_index, basct::span<Element> products,
                     basct::blob_array& output_digit_or_all, bool is_signed) noexcept {
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
  SXT_STACK_ARRAY(pos_products, pos_digit_count_one, Element);
  std::copy_n(output_products.begin(), pos_digit_count_one, pos_products.begin());
  auto neg_products = products.subspan(input_index + pos_digit_count_one, neg_digit_count_one);
  for (auto& e : neg_products) {
    neg(e, e);
  }
  size_t pos_index = 0;
  size_t neg_index = 0;
  size_t out_index = 0;
  basbt::for_each_bit(digit_or_all, [&](size_t bit_index) noexcept {
    auto is_pos_set = basbt::test_bit(pos_digit_or_all, bit_index);
    auto is_neg_set = basbt::test_bit(neg_digit_or_all, bit_index);
    if (is_pos_set && is_neg_set) {
      add(output_products[out_index++], pos_products[pos_index++], neg_products[neg_index++]);
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
// fold_multiproducts
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
void fold_multiproducts(memmg::managed_array<Element>& products, basct::span<uint8_t> digit_or_all,
                        basct::cspan<Element> products_p,
                        basct::cspan<uint8_t> digit_or_all_p) noexcept {
  auto num_bytes = digit_or_all.size();
  // clang-format off
  SXT_DEBUG_ASSERT(
      digit_or_all.size() == num_bytes &&
      basbt::pop_count(digit_or_all) == products.size() &&
      digit_or_all_p.size() == num_bytes &&
      basbt::pop_count(digit_or_all_p) == products_p.size()
  );
  // clang-format on
  SXT_STACK_ARRAY(digit_or_all_pp, num_bytes, uint8_t);
  std::copy(digit_or_all.begin(), digit_or_all.end(), digit_or_all_pp.begin());
  basbt::or_equal(digit_or_all_pp, digit_or_all_p);
  auto bit_count = basbt::pop_count(digit_or_all_pp);
  size_t index = 0;
  size_t index_p = 0;
  size_t index_pp = 0;
  memmg::managed_array<Element> products_pp(bit_count, products.get_allocator());
  basbt::for_each_bit(digit_or_all_pp, [&](size_t bit_index) noexcept {
    auto is_set = basbt::test_bit(digit_or_all, bit_index);
    auto is_set_p = basbt::test_bit(digit_or_all_p, bit_index);
    SXT_DEBUG_ASSERT(is_set || is_set_p);
    if (is_set && is_set_p) {
      add(products_pp[index_pp++], products[index++], products_p[index_p++]);
    } else if (is_set) {
      products_pp[index_pp++] = products[index++];
    } else {
      products_pp[index_pp++] = products_p[index_p++];
    }
  });
  std::copy(digit_or_all_pp.begin(), digit_or_all_pp.end(), digit_or_all.begin());
  products = std::move(products_pp);
}

//--------------------------------------------------------------------------------------------------
// combine_multiproducts
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
void combine_multiproducts(basct::span<Element> outputs,
                           const basct::blob_array& output_digit_or_all,
                           basct::cspan<Element> products) noexcept {
  size_t input_index = 0;
  for (size_t output_index = 0; output_index < output_digit_or_all.size(); ++output_index) {
    auto digit_or_all = output_digit_or_all[output_index];
    int digit_count_one = basbt::pop_count(digit_or_all);
    if (digit_count_one == 0) {
      outputs[output_index] = Element::identity();
      continue;
    }
    Element output;
    SXT_DEBUG_ASSERT(input_index + digit_count_one <= products.size());
    doubling_reduce(
        output, digit_or_all,
        basct::cspan<Element>{&products[input_index], static_cast<size_t>(digit_count_one)});
    input_index += digit_count_one;
    outputs[output_index] = output;
  }
}

template <bascrv::element Element>
void combine_multiproducts(basct::span<Element> outputs, basct::blob_array& output_digit_or_all,
                           basct::span<Element> products,
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
        init_output_products<Element>(product_index, input_index, products, output_digit_or_all,
                                      static_cast<bool>(sequence.is_signed));
    if (output_products.empty()) {
      outputs[sequence_index] = Element::identity();
      continue;
    }
    doubling_reduce(outputs[sequence_index], digit_or_all, output_products);
  }
}
} // namespace sxt::mtxcrv
