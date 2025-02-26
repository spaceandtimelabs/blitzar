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
#include "sxt/base/field/element.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// sumcheck_transcript
//--------------------------------------------------------------------------------------------------
template <basfld::element T> class sumcheck_transcript {
public:
  virtual ~sumcheck_transcript() noexcept = default;

  virtual void init(size_t num_variables, size_t round_degree) noexcept = 0;

  virtual void round_challenge(T& r, basct::cspan<T> polynomial) noexcept = 0;
};
} // namespace sxt::prfsk2
