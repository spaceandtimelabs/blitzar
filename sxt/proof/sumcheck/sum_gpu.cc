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
#include "sxt/proof/sumcheck/sum_gpu.h"

#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum
//--------------------------------------------------------------------------------------------------
void sum(basct::span<s25t::element> polynomial, basdv::stream& stream,
         basct::cspan<s25t::element> mles,
         basct::cspan<std::pair<s25t::element, unsigned>> product_table,
         basct::cspan<unsigned> product_terms, unsigned mid, unsigned n) noexcept {
  (void)polynomial;
  (void)stream;
  (void)mles;
  (void)product_table;
  (void)product_terms;
  (void)mid;
  (void)n;
}
} // namespace sxt::prfsk