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
#include "sxt/scalar25/property/zero.h"

#include "sxt/base/bit/zero_equality.h"
#include "sxt/scalar25/base/reduce.h"

namespace sxt::s25p {
//--------------------------------------------------------------------------------------------------
// is_zero
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
int is_zero(const s25t::element& e) noexcept {
  s25t::element t = e;

  // this `reduce` is necessary for non-reduced values,
  // since `is_zero` do not detect them
  // Ex: 2^252 + 27742317777372353535851937790883648493
  // will not be considered zero by `is_zero`
  s25b::reduce32(t.data());

  return basbt::is_zero(t.data(), sizeof(t));
}
} // namespace sxt::s25p
