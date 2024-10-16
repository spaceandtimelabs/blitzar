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

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum_round
//--------------------------------------------------------------------------------------------------
template <size_t MaxDegree, class Scalar>
void sum_round(Scalar* __restrict__ polynomial, const Scalar* __restrict__ mles,
               const unsigned* __restrict__ product_table,
               const unsigned* __restrict__ product_lengths, unsigned num_products) noexcept {
  (void)polynomial;
  (void)mles;
  (void)product_table;
  (void)product_lengths;
  (void)num_products;
}
} // namespace sxt::prfsk
