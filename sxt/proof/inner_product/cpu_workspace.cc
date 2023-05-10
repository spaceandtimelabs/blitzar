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
#include "sxt/proof/inner_product/cpu_workspace.h"

#include "sxt/execution/async/future.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// ap_value
//--------------------------------------------------------------------------------------------------
xena::future<void> cpu_workspace::ap_value(s25t::element& value) const noexcept {
  value = this->a_vector[0];
  return xena::make_ready_future();
}
} // namespace sxt::prfip
