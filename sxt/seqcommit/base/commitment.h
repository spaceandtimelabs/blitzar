#pragma once

#include <cstdint>
#include <initializer_list>
#include <iosfwd>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::sqcb {
//--------------------------------------------------------------------------------------------------
// commitment
//--------------------------------------------------------------------------------------------------
class commitment {
 public:
   commitment() noexcept = default;

   commitment(std::initializer_list<uint8_t> values) noexcept;

   CUDA_CALLABLE
   uint8_t* data() noexcept { return data_; }

   CUDA_CALLABLE
   const uint8_t* data() const noexcept { return data_; }
 private:
  uint8_t data_[32]= {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };
};

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const commitment& lhs, const commitment& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
inline bool operator!=(const commitment& lhs, const commitment& rhs) noexcept {
  return !(lhs == rhs);
}

//--------------------------------------------------------------------------------------------------
// opeator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const commitment& c) noexcept;
} // namespace sxt::sqcb
