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
#include "sxt/proof/inner_product/generator_fold.h"

#include <algorithm>
#include <cstring>

#include "sxt/base/error/assert.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/scalar25/constant/max_bits.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// decompose_generator_fold
//--------------------------------------------------------------------------------------------------
void decompose_generator_fold(basct::span<unsigned>& res, const s25t::element& m_low,
                              const s25t::element& m_high) noexcept {
  SXT_DEBUG_ASSERT(res.size() == s25cn::max_bits_v);
  size_t bit_index = 0;
  for (size_t i = 0; i < 4; ++i) {
    uint64_t x, y;
    std::memcpy(&x, reinterpret_cast<const char*>(&m_low) + sizeof(uint64_t) * i, sizeof(uint64_t));
    std::memcpy(&y, reinterpret_cast<const char*>(&m_high) + sizeof(uint64_t) * i,
                sizeof(uint64_t));
    auto m = std::min(s25cn::max_bits_v - bit_index, 64ul);
    for (size_t bit_offset = 0; bit_offset < m; ++bit_offset) {
      auto mask = 1ull << bit_index;
      res[bit_index++] =
          static_cast<unsigned>((x & mask) != 0) + 2u * static_cast<unsigned>((y & mask) != 0);
    }
  }
  size_t size = s25cn::max_bits_v;
  while (size > 0) {
    if (res[size - 1] != 0) {
      break;
    }
    --size;
  }
  res = res.subspan(0, size);
}

//--------------------------------------------------------------------------------------------------
// fold_generators
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void fold_generators(c21t::element_p3& res, basct::cspan<unsigned> decomposition,
                                   const c21t::element_p3& g_low,
                                   const c21t::element_p3& g_high) noexcept {
  if (decomposition.empty()) {
    // this should never happen
    res = c21cn::zero_p3_v;
    return;
  }
  c21t::element_p3 terms[3];
  terms[0] = g_low;
  terms[1] = g_high;
  c21o::add(terms[2], g_low, g_high);

  size_t bit_index = decomposition.size();
  --bit_index;
  assert(0 < decomposition[bit_index] && decomposition[bit_index] < 4);
  res = terms[decomposition[bit_index] - 1];

  while (bit_index > 0) {
    auto term_index = decomposition[--bit_index];
    c21o::double_element(res, res);
    assert(term_index < 4);
    if (term_index == 0) {
      continue;
    }
    c21o::add(res, res, terms[term_index - 1]);
  }
}
} // namespace sxt::prfip
