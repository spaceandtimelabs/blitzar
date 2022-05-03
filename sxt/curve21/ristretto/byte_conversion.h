#pragma once

#include "sxt/base/macro/cuda_callable.h"

#include <stdint.h>

namespace sxt::c21t { struct element_p3; }
namespace sxt::f51t { class element; }

namespace sxt::c21rs {

//--------------------------------------------------------------------------------------------------
// byte_conversion
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void to_bytes(uint8_t s[32], const c21t::element_p3& p) noexcept;

}  // namespace sxt::c21rs
