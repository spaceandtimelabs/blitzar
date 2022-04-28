#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t { struct element_p3; }

namespace sxt {
//--------------------------------------------------------------------------------------------------
// multiply_add
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void multiply_add(c21t::element_p3& res, int mi, int i) noexcept;
} // namespace sxt
