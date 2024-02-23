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
#include "sxt/ristretto/operation/scalar_multiply.h"

#include "sxt/curve32/operation/scalar_multiply.h"
#include "sxt/curve32/type/element_p3.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::rsto {
//--------------------------------------------------------------------------------------------------
// scalar_multiply
//--------------------------------------------------------------------------------------------------
/*
 h = a * p
 where a = a[0]+256*a[1]+...+256^31 a[31]
 */
CUDA_CALLABLE
void scalar_multiply(rstt::compressed_element& r, basct::cspan<uint8_t> a,
                     const rstt::compressed_element& p) noexcept {

  c32t::element_p3 temp_p;

  rstb::from_bytes(temp_p, p.data());

  c32o::scalar_multiply(temp_p, a, temp_p);

  rstb::to_bytes(r.data(), temp_p);
}
} // namespace sxt::rsto
