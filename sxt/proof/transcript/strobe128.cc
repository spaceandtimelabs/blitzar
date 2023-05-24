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
#include "sxt/proof/transcript/strobe128.h"

#include "sxt/base/error/assert.h"
#include "sxt/proof/transcript/keccakf.h"

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// strobe_r_v
//--------------------------------------------------------------------------------------------------
static constexpr uint8_t strobe_r_v = 166;

//--------------------------------------------------------------------------------------------------
// flag_i_v
//--------------------------------------------------------------------------------------------------
static constexpr uint8_t flag_i_v = 1;

//--------------------------------------------------------------------------------------------------
// flag_a_v
//--------------------------------------------------------------------------------------------------
static constexpr uint8_t flag_a_v = 1 << 1;

//--------------------------------------------------------------------------------------------------
// flag_c_v
//--------------------------------------------------------------------------------------------------
static constexpr uint8_t flag_c_v = 1 << 2;

//--------------------------------------------------------------------------------------------------
// flag_t_v
//--------------------------------------------------------------------------------------------------
static constexpr uint8_t flag_t_v = 1 << 3;

//--------------------------------------------------------------------------------------------------
// flag_m_v
//--------------------------------------------------------------------------------------------------
static constexpr uint8_t flag_m_v = 1 << 4;

//--------------------------------------------------------------------------------------------------
// flag_k_v
//--------------------------------------------------------------------------------------------------
static constexpr uint8_t flag_k_v = 1 << 5;

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
strobe128::strobe128(std::string_view label) noexcept {
  keccakf(state_bytes_);
  meta_ad({reinterpret_cast<const uint8_t*>(label.data()), label.size()}, false);
}

//--------------------------------------------------------------------------------------------------
// meta_ad
//--------------------------------------------------------------------------------------------------
void strobe128::meta_ad(basct::cspan<uint8_t> data, bool more) noexcept {
  begin_op(flag_m_v | flag_a_v, more);
  absorb(data);
}

//--------------------------------------------------------------------------------------------------
// ad
//--------------------------------------------------------------------------------------------------
void strobe128::ad(basct::cspan<uint8_t> data, bool more) noexcept {
  begin_op(flag_a_v, more);
  absorb(data);
}

//--------------------------------------------------------------------------------------------------
// prf
//--------------------------------------------------------------------------------------------------
void strobe128::prf(basct::span<uint8_t> data, bool more) noexcept {
  begin_op(flag_i_v | flag_a_v | flag_c_v, more);
  squeeze(data);
}

//--------------------------------------------------------------------------------------------------
// key
//--------------------------------------------------------------------------------------------------
void strobe128::key(basct::cspan<uint8_t> data, bool more) noexcept {
  begin_op(flag_a_v | flag_c_v, more);
  overwrite(data);
}

//--------------------------------------------------------------------------------------------------
// run_f
//--------------------------------------------------------------------------------------------------
void strobe128::run_f() noexcept {
  state_bytes_[pos_] ^= pos_begin_;
  state_bytes_[pos_ + 1] ^= 0x04;
  state_bytes_[strobe_r_v + 1] ^= 0x80;

  keccakf(state_bytes_);

  pos_ = 0;
  pos_begin_ = 0;
}

//--------------------------------------------------------------------------------------------------
// absorb
//--------------------------------------------------------------------------------------------------
void strobe128::absorb(basct::cspan<uint8_t> data) noexcept {
  for (size_t i = 0; i < data.size(); ++i) {
    state_bytes_[pos_] ^= data[i];
    pos_ += 1;
    if (pos_ == strobe_r_v) {
      run_f();
    }
  }
}

//--------------------------------------------------------------------------------------------------
// begin_op
//--------------------------------------------------------------------------------------------------
void strobe128::begin_op(uint8_t flags, bool more) noexcept {
  if (more) {
    /* Changing flags while continuing is illegal */
    SXT_DEBUG_ASSERT(cur_flags_ == flags);
    return;
  }

  /* T flag is not supported */
  SXT_DEBUG_ASSERT(!(flags & flag_t_v));

  uint8_t old_begin = pos_begin_;
  pos_begin_ = pos_ + 1;
  cur_flags_ = flags;

  uint8_t data[2] = {old_begin, flags};
  absorb({data, 2});

  /* Force running the permutation if C or K is set. */
  uint8_t force_f = 0 != (flags & (flag_c_v | flag_k_v));

  if (force_f && pos_ != 0) {
    run_f();
  }
}

//--------------------------------------------------------------------------------------------------
// squeeze
//--------------------------------------------------------------------------------------------------
void strobe128::squeeze(basct::span<uint8_t> data) noexcept {
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = state_bytes_[pos_];
    state_bytes_[pos_] = 0;
    pos_ += 1;
    if (pos_ == strobe_r_v) {
      run_f();
    }
  }
}

//--------------------------------------------------------------------------------------------------
// overwrite
//--------------------------------------------------------------------------------------------------
void strobe128::overwrite(basct::cspan<uint8_t> data) noexcept {
  for (size_t i = 0; i < data.size(); ++i) {
    state_bytes_[pos_] = data[i];
    pos_ += 1;
    if (pos_ == strobe_r_v) {
      run_f();
    }
  }
}
} // namespace sxt::prft
