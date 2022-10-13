#pragma once

#include <cstdint>
#include <initializer_list>
#include <iosfwd>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::s25t {
//--------------------------------------------------------------------------------------------------
// element
//--------------------------------------------------------------------------------------------------
/**
 * non-reduced elements should be in the [0..L) interval,
 * L being the order of the main subgroup
 * (L = 2^252 + 27742317777372353535851937790883648493).
 */
class element {
public:
  element() noexcept = default;

  explicit element(std::initializer_list<uint8_t> values) noexcept;

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
bool operator==(const element& lhs, const element& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
inline bool operator!=(const element& lhs, const element& rhs) noexcept { return !(lhs == rhs); }

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const element& c) noexcept;
} // namespace sxt::s25t
