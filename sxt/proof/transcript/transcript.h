#pragma once

#include <cassert>
#include <cstdint>

#include "sxt/base/container/span.h"
#include "sxt/proof/transcript/strobe128.h"

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// transcript
//--------------------------------------------------------------------------------------------------
class transcript {
public:
  explicit transcript(basct::cspan<uint8_t> label) noexcept;

  void append_message(basct::cspan<uint8_t> label, basct::cspan<uint8_t> message) noexcept;

  void challenge_bytes(basct::span<uint8_t> dest, basct::cspan<uint8_t> label) noexcept;

private:
  strobe128 strobe_;
};
} // namespace sxt::prft
