#pragma once

#include <array>
#include <cstdint>
#include <iosfwd>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::f51t {
//--------------------------------------------------------------------------------------------------
// element
//--------------------------------------------------------------------------------------------------
class element {
 public:
  element() noexcept = default;

  CUDA_CALLABLE constexpr element(uint64_t x1, uint64_t x2, uint64_t x3,
                                  uint64_t x4, uint64_t x5) noexcept
      : data_{x1, x2, x3, x4, x5} {}

  CUDA_CALLABLE constexpr const uint64_t& operator[](int index) const noexcept {
    return data_[index];
  }

  CUDA_CALLABLE constexpr uint64_t& operator[](int index) noexcept { return data_[index]; }

  CUDA_CALLABLE constexpr const uint64_t* data() const noexcept { return data_; }

  CUDA_CALLABLE constexpr uint64_t* data() noexcept { return data_; }

 private:
   uint64_t data_[5];
};

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const element& e) noexcept;

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const element& lhs, const element& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
inline bool operator!=(const element& lhs, const element& rhs) noexcept {
  return !(lhs == rhs);
}
}  // namespace sxt::f51t
