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

#include <cstdint>
#include <string>

#include "sxt/base/container/span.h"

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// strobe128
//--------------------------------------------------------------------------------------------------
class strobe128 {
public:
  explicit strobe128(std::string_view label) noexcept;

  void meta_ad(basct::cspan<uint8_t> data, bool more) noexcept;

  void ad(basct::cspan<uint8_t> data, bool more) noexcept;

  void prf(basct::span<uint8_t> data, bool more) noexcept;

  void key(basct::cspan<uint8_t> data, bool more) noexcept;

private:
  uint8_t state_bytes_[200] = {1,  168, 1,   0,  1,  96, 83, 84, 82, 79,
                               66, 69,  118, 49, 46, 48, 46, 50, 0};
  uint8_t pos_ = 0;
  uint8_t pos_begin_ = 0;
  uint8_t cur_flags_ = 0;

  void run_f() noexcept;
  void absorb(basct::cspan<uint8_t> data) noexcept;
  void begin_op(uint8_t flags, bool more) noexcept;
  void squeeze(basct::span<uint8_t> data) noexcept;
  void overwrite(basct::cspan<uint8_t> data) noexcept;
};
} // namespace sxt::prft
