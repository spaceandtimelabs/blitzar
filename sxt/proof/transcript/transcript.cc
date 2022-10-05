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

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// encode_usize_as_u32
//--------------------------------------------------------------------------------------------------
static uint32_t encode_usize_as_u32(size_t len) noexcept {
  assert(len <= (static_cast<size_t>(UINT32_MAX)));
  return static_cast<uint32_t>(len);
}

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
transcript::transcript(basct::cspan<uint8_t> label) noexcept
    : strobe_({reinterpret_cast<const uint8_t*>("Merlin v1.0"), 11}) {
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
  append_message({reinterpret_cast<const uint8_t*>("dom-sep"), 7}, label);
}

//--------------------------------------------------------------------------------------------------
// append_message
//--------------------------------------------------------------------------------------------------
void transcript::append_message(basct::cspan<uint8_t> label,
                                basct::cspan<uint8_t> message) noexcept {
  /// Append a prover's `message` to the transcript.
  ///
  /// The `label` parameter is metadata about the message, and is
  /// also appended to the transcript. See the [Transcript
  /// Protocols](https://merlin.cool/use/protocol.html) section of
  /// the Merlin website for details on labels.
  const uint32_t data_len = encode_usize_as_u32(message.size());
  strobe_.meta_ad(label, false);
  strobe_.meta_ad({reinterpret_cast<const uint8_t*>(&data_len), 4}, true);
  strobe_.ad(message, false);
}

//--------------------------------------------------------------------------------------------------
// challenge_bytes
//--------------------------------------------------------------------------------------------------
void transcript::challenge_bytes(basct::span<uint8_t> dest, basct::cspan<uint8_t> label) noexcept {
  /// Fill the supplied buffer with the verifier's challenge bytes.
  ///
  /// The `label` parameter is metadata about the challenge, and is
  /// also appended to the transcript. See the [Transcript
  /// Protocols](https://merlin.cool/use/protocol.html) section of
  /// the Merlin website for details on labels.
  const uint32_t data_len = encode_usize_as_u32(dest.size());
  strobe_.meta_ad(label, false);
  strobe_.meta_ad({reinterpret_cast<const uint8_t*>(&data_len), 4}, true);
  strobe_.prf(dest, false);
}
} // namespace sxt::prft
