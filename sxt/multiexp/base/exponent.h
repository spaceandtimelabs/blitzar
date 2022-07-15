#pragma once

#include <cstdint>

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// exponent
//--------------------------------------------------------------------------------------------------
class exponent {
public:
  exponent() noexcept = default;

  exponent(uint64_t value1, uint64_t value2, uint64_t value3, uint64_t value4) noexcept;

  const uint8_t* data() const noexcept { return reinterpret_cast<const uint8_t*>(data_); }

  uint8_t* data() noexcept { return reinterpret_cast<uint8_t*>(data_); }

  bool operator<(const exponent& rhs) const noexcept;

  int highest_bit() const noexcept;

private:
  uint64_t data_[4] = {};
};

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const exponent& lhs, const exponent& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
inline bool operator!=(const exponent& lhs, const exponent& rhs) noexcept { return !(lhs == rhs); }
} // namespace sxt::mtxb
