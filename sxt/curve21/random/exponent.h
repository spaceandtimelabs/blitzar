#pragma once

#include <cstring>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/fast_random_number_generator.h"

namespace sxt::c21rn {
//--------------------------------------------------------------------------------------------------
// generate_random_exponent
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void generate_random_exponent(unsigned char a[32],
                                     basn::fast_random_number_generator& generator) noexcept {
  for (int i = 0; i < 32; i += 8) {
    auto x = generator();
    std::memcpy(static_cast<void*>(a + i), static_cast<void*>(&x), sizeof(x));
  }
  // make sure a[31] <= 127
  a[31] = a[31] >> 1;
}
} // namespace sxt::c21rn
