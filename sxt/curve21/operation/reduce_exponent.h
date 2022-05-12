#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21o {

//--------------------------------------------------------------------------------------------------
// reduce_exponent
//--------------------------------------------------------------------------------------------------
// s = s % p where
// p = 2^252 + 27742317777372353535851937790883648493
CUDA_CALLABLE
void reduce_exponent(unsigned char s[32]) noexcept;
}  // namespace sxt::c21o
