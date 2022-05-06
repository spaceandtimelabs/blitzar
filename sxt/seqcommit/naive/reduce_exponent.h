#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::sqcnv {

//--------------------------------------------------------------------------------------------------
// reduce_exponent
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void reduce_exponent(unsigned char s[32]) noexcept;

}  // namespace sxt::sqcnv
