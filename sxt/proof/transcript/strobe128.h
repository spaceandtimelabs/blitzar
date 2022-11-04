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
