#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::rstt {
struct compressed_element;
}

namespace sxt::rsto {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/*
 r = p + q
 */
CUDA_CALLABLE
void add(rstt::compressed_element& r, const rstt::compressed_element& p,
         const rstt::compressed_element& q) noexcept;
} // namespace sxt::rsto
