#pragma once

#include <cstdint>
#include <string>

#include "sxt/base/container/span.h"
#include "sxt/proof/transcript/strobe128.h"

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// transcript
//--------------------------------------------------------------------------------------------------
class transcript {
public:
  explicit transcript(std::string_view label) noexcept;

  void append_message(std::string_view label, basct::cspan<uint8_t> message) noexcept;

  void challenge_bytes(basct::span<uint8_t> dest, std::string_view label) noexcept;

private:
  strobe128 strobe_;
};

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const transcript& lhs, const transcript& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
inline bool operator!=(const transcript& lhs, const transcript& rhs) noexcept {
  return !(lhs == rhs);
}
} // namespace sxt::prft
