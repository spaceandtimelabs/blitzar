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
/** libmerlin
 *
 * An implementation of Merlin transcripts.
 *
 * Original Author: Henry de Valence <hdevalence@hdevalence.ca>
 * Modified by: Jose Ribeiro <joseribeiro1017@gmail.com>
 *
 * Derived from keccak-tiny, with attribution note preserved below:
 *
 * Implementor: David Leon Gil
 * License: CC0, attribution kindly requested. Blame taken too,
 * but not liability.
 *
 * See third_party/license/keccak-tiny.LICENSE
 */
#include "sxt/proof/transcript/transcript.h"

#include <cstring>

#include "sxt/base/error/assert.h"

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// encode_usize_as_u32
//--------------------------------------------------------------------------------------------------
static uint32_t encode_usize_as_u32(size_t len) noexcept {
  SXT_DEBUG_ASSERT(len <= (static_cast<size_t>(UINT32_MAX)));
  return static_cast<uint32_t>(len);
}

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
transcript::transcript(std::string_view label) noexcept : strobe_("Merlin v1.0") {
  /// Initialize a new transcript with the supplied `label`, which
  /// is used as a domain separator.
  ///
  /// # Note
  ///
  /// This function should be called by a proof library's API
  /// consumer (i.e., the application using the proof library), and
  /// **not by the proof implementation**. See the [Passing
  /// Transcripts](https://merlin.cool/use/passing.html) section of
  /// the Merlin website for more details on why.
  append_message("dom-sep", {reinterpret_cast<const uint8_t*>(label.data()), label.size()});
}

//--------------------------------------------------------------------------------------------------
// append_message
//--------------------------------------------------------------------------------------------------
void transcript::append_message(std::string_view label, basct::cspan<uint8_t> message) noexcept {
  /// Append a prover's `message` to the transcript.
  ///
  /// The `label` parameter is metadata about the message, and is
  /// also appended to the transcript. See the [Transcript
  /// Protocols](https://merlin.cool/use/protocol.html) section of
  /// the Merlin website for details on labels.
  const uint32_t data_len = encode_usize_as_u32(message.size());
  strobe_.meta_ad({reinterpret_cast<const uint8_t*>(label.data()), label.size()}, false);
  strobe_.meta_ad({reinterpret_cast<const uint8_t*>(&data_len), sizeof(uint32_t)}, true);
  strobe_.ad(message, false);
}

//--------------------------------------------------------------------------------------------------
// challenge_bytes
//--------------------------------------------------------------------------------------------------
void transcript::challenge_bytes(basct::span<uint8_t> dest, std::string_view label) noexcept {
  /// Fill the supplied buffer with the verifier's challenge bytes.
  ///
  /// The `label` parameter is metadata about the challenge, and is
  /// also appended to the transcript. See the [Transcript
  /// Protocols](https://merlin.cool/use/protocol.html) section of
  /// the Merlin website for details on labels.
  const uint32_t data_len = encode_usize_as_u32(dest.size());
  strobe_.meta_ad({reinterpret_cast<const uint8_t*>(label.data()), label.size()}, false);
  strobe_.meta_ad({reinterpret_cast<const uint8_t*>(&data_len), sizeof(uint32_t)}, true);
  strobe_.prf(dest, false);
}

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const transcript& lhs, const transcript& rhs) noexcept {
  return std::memcmp(static_cast<const void*>(&lhs), static_cast<const void*>(&rhs),
                     sizeof(transcript)) == 0;
}
} // namespace sxt::prft
