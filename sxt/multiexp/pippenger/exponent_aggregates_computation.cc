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
#include "sxt/multiexp/pippenger/exponent_aggregates_computation.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "sxt/base/bit/count.h"
#include "sxt/base/bit/span_op.h"
#include "sxt/base/num/abs.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/constexpr_switch.h"
#include "sxt/base/num/power2_equality.h"
#include "sxt/base/type/int.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/pippenger/exponent_aggregates.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// aggegate_term
//--------------------------------------------------------------------------------------------------
static void aggegate_term(exponent_aggregates& aggregates, basct::cspan<uint8_t> term,
                          size_t output_index, size_t term_index) noexcept {
  basbt::or_equal(aggregates.term_or_all[term_index], term);
  basbt::or_equal(aggregates.output_or_all[output_index], term);
  basbt::max_equal(aggregates.max_exponent, term);
  aggregates.pop_count += basbt::pop_count(term);
}

//--------------------------------------------------------------------------------------------------
// aggregate_unsigned_terms
//--------------------------------------------------------------------------------------------------
static void aggregate_unsigned_terms(exponent_aggregates& aggregates, size_t output_index,
                                     const mtxb::exponent_sequence& sequence) noexcept {
  auto element_nbytes = sequence.element_nbytes;
  for (size_t term_index = 0; term_index < sequence.n; ++term_index) {
    basct::cspan<uint8_t> term{sequence.data + term_index * element_nbytes, element_nbytes};
    aggegate_term(aggregates, term, output_index, term_index);
  }
}

//--------------------------------------------------------------------------------------------------
// aggregate_signed_terms
//--------------------------------------------------------------------------------------------------
template <size_t NumBytes>
static void aggregate_signed_terms(exponent_aggregates& aggregates, size_t output_index,
                                   const mtxb::exponent_sequence& sequence) noexcept {
  for (size_t term_index = 0; term_index < sequence.n; ++term_index) {
    bast::sized_int_t<NumBytes * 8> x;
    std::memcpy(reinterpret_cast<uint8_t*>(&x), sequence.data + term_index * NumBytes, NumBytes);
    auto abs_x = basn::abs(x);
    basct::cspan<uint8_t> term{reinterpret_cast<uint8_t*>(&abs_x), NumBytes};
    if (x == abs_x) {
      aggegate_term(aggregates, term, output_index, term_index);
    } else {
      aggegate_term(aggregates, term, output_index + 1, term_index);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// compute_exponent_aggregates
//--------------------------------------------------------------------------------------------------
void compute_exponent_aggregates(exponent_aggregates& aggregates,
                                 basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  size_t max_sequence_length = 0;
  uint8_t max_element_nbytes = 0;
  size_t output_count = 0;
  for (auto& sequence : exponents) {
    max_sequence_length = std::max(max_sequence_length, sequence.n);
    max_element_nbytes = std::max(max_element_nbytes, sequence.element_nbytes);
    output_count += 1 + sequence.is_signed;
  }
  aggregates.max_exponent.resize(max_element_nbytes);
  aggregates.term_or_all.resize(max_sequence_length, max_element_nbytes);
  aggregates.output_or_all.resize(output_count, max_element_nbytes);
  aggregates.pop_count = 0;

  size_t output_index = 0;
  for (size_t sequence_index = 0; sequence_index < exponents.size(); ++sequence_index) {
    auto sequence = exponents[sequence_index];
    auto element_num_bytes = sequence.element_nbytes;
    if (sequence.is_signed) {
      SXT_DEBUG_ASSERT(basn::is_power2(element_num_bytes));
      SXT_RELEASE_ASSERT(element_num_bytes <= 16,
                         "signed commitments for numbers larger than 128-bits aren't supported");
      basn::constexpr_switch<5>(
          basn::ceil_log2(element_num_bytes),
          [&]<unsigned NumBytesLg2>(std::integral_constant<unsigned, NumBytesLg2>) noexcept {
            static constexpr auto NumBytes = 1ull << NumBytesLg2;
            aggregate_signed_terms<NumBytes>(aggregates, output_index, sequence);
          });
    } else {
      aggregate_unsigned_terms(aggregates, output_index, sequence);
    }
    output_index += 1 + sequence.is_signed;
  }
}
} // namespace sxt::mtxpi
