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

#include <utility>

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::prft {
class transcript;
}
namespace sxt::s25t {
class element;
}

namespace sxt::prfsk {
class driver;
class sumcheck_transcript;

//--------------------------------------------------------------------------------------------------
// prove_sum
//--------------------------------------------------------------------------------------------------
xena::future<> prove_sum(basct::span<s25t::element> polynomials,
                         basct::span<s25t::element> evaluation_point,
                         sumcheck_transcript& transcript, const driver& drv,
                         basct::cspan<s25t::element> mles,
                         basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                         basct::cspan<unsigned> product_terms, unsigned n) noexcept;
} // namespace sxt::prfsk
