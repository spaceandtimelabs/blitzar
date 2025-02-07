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
#include "sxt/proof/sumcheck/reference_transcript.h"

#include "sxt/proof/transcript/transcript_utility.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
reference_transcript::reference_transcript(prft::transcript& transcript) noexcept
    : transcript_{transcript} {}

//--------------------------------------------------------------------------------------------------
// init
//--------------------------------------------------------------------------------------------------
void reference_transcript::init(size_t num_variables, size_t round_degree) noexcept {
  prft::set_domain(transcript_, "sumcheck proof v1");
  prft::append_value(transcript_, "n", num_variables);
  prft::append_value(transcript_, "k", round_degree);
}

//--------------------------------------------------------------------------------------------------
// round_challenge
//--------------------------------------------------------------------------------------------------
void reference_transcript::round_challenge(s25t::element& r,
                                           basct::cspan<s25t::element> polynomial) noexcept {
  prft::append_values(transcript_, "P", polynomial);
  prft::challenge_value(r, transcript_, "R");
}
} // namespace sxt::prfsk
