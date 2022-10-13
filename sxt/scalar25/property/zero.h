#pragma once

#include "sxt/scalar25/type/element.h"

namespace sxt::s25p {
//--------------------------------------------------------------------------------------------------
// is_zero
//--------------------------------------------------------------------------------------------------
// return 1 if e % L = 0 (L = 2^252 + 27742317777372353535851937790883648493)
//        0 otherwise
CUDA_CALLABLE
int is_zero(const s25t::element& e) noexcept;
} // namespace sxt::s25p
