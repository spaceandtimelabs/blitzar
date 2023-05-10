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
#pragma once

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::rstt {
class compressed_element;
}
namespace sxt::s25t {
struct element;
}
namespace sxt::prft {
class transcript;
}

namespace sxt::prfip {
class driver;
struct proof_descriptor;

//--------------------------------------------------------------------------------------------------
// prove_inner_product
//--------------------------------------------------------------------------------------------------
xena::future<void> prove_inner_product(basct::span<rstt::compressed_element> l_vector,
                                       basct::span<rstt::compressed_element> r_vector,
                                       s25t::element& ap_value, prft::transcript& transcript,
                                       const driver& drv, const proof_descriptor& descriptor,
                                       basct::cspan<s25t::element> a_vector) noexcept;

//--------------------------------------------------------------------------------------------------
// verify_inner_product
//--------------------------------------------------------------------------------------------------
xena::future<bool> verify_inner_product(prft::transcript& transcript, const driver& drv,
                                        const proof_descriptor& descriptor,
                                        const s25t::element& product,
                                        const c21t::element_p3& a_commit,
                                        basct::cspan<rstt::compressed_element> l_vector,
                                        basct::cspan<rstt::compressed_element> r_vector,
                                        const s25t::element& ap_value) noexcept;
} // namespace sxt::prfip
