/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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

namespace sxt::basdv {
class stream;
}
namespace sxt::s25t {
class element;
}

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// reduce_sums
//--------------------------------------------------------------------------------------------------
xena::future<> reduce_sums(basct::span<s25t::element> p, basdv::stream& stream,
                           basct::cspan<s25t::element> partial_terms) noexcept;
} // namespace sxt::prfsk
