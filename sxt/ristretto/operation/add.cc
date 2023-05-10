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
#include "sxt/ristretto/operation/add.h"

#include "sxt/curve21/operation/add.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::rsto {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/*
 r = p + q
 */
CUDA_CALLABLE
void add(rstt::compressed_element& r, const rstt::compressed_element& p,
         const rstt::compressed_element& q) noexcept {

  c21t::element_p3 temp_p, temp_q;

  rstb::from_bytes(temp_p, p.data());
  rstb::from_bytes(temp_q, q.data());

  c21o::add(temp_p, temp_p, temp_q);

  rstb::to_bytes(r.data(), temp_p);
}
} // namespace sxt::rsto
