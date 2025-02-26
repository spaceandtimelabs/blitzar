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

#include "sxt/proof/sumcheck2/sumcheck_transcript.h"

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// callback_sumcheck_transcript
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
class callback_sumcheck_transcript final : public prfsk2::sumcheck_transcript<T> {
public:
  using callback_t = void (*)(T* r, void* context, const T* polynomial, unsigned polynomial_len);

  callback_sumcheck_transcript(callback_t f, void* context) noexcept : f_{f}, context_{context} {}

  void init(size_t /*num_variables*/, size_t /*round_degree*/) noexcept override {}

  void round_challenge(T& r, basct::cspan<T> polynomial) noexcept override {
    f_(&r, context_, polynomial.data(), static_cast<unsigned>(polynomial.size()));
  }

private:
  callback_t f_;
  void* context_;
};
} // namespace sxt::cbnbck
