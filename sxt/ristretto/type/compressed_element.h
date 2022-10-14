#pragma once

#include <cstdint>
#include <initializer_list>
#include <iosfwd>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::rstt {
//--------------------------------------------------------------------------------------------------
// compressed_element
//--------------------------------------------------------------------------------------------------
class compressed_element {
public:
  compressed_element() noexcept = default;

  explicit compressed_element(std::initializer_list<uint8_t> values) noexcept;

  CUDA_CALLABLE
  uint8_t* data() noexcept { return data_; }

  CUDA_CALLABLE
  const uint8_t* data() const noexcept { return data_; }

private:
  uint8_t data_[32] = {};
};

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const compressed_element& lhs, const compressed_element& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
inline bool operator!=(const compressed_element& lhs, const compressed_element& rhs) noexcept {
  return !(lhs == rhs);
}

//--------------------------------------------------------------------------------------------------
// opeator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const compressed_element& c) noexcept;
} // namespace sxt::rstt
