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
#include "sxt/proof/sumcheck/transcript_utility.h"

#include "sxt/proof/transcript/transcript_utility.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// init_transcript
//--------------------------------------------------------------------------------------------------
void init_transcript(prft::transcript& transcript, unsigned num_variables,
                     unsigned round_degree) noexcept {
  prft::set_domain(transcript, "sumcheck proof v1");
  prft::append_value(transcript, "n", num_variables);
  prft::append_value(transcript, "k", round_degree);
}

//--------------------------------------------------------------------------------------------------
// round_challenge
//--------------------------------------------------------------------------------------------------
void round_challenge(s25t::element& r, prft::transcript& transcript,
                     basct::cspan<s25t::element> polynomial) noexcept {
  prft::append_values(transcript, "P", polynomial);
  prft::challenge_value(r, transcript, "R");
}
} // namespace sxt::prfsk
