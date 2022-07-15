#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::f51t {
class element;
}

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// pow22523
//--------------------------------------------------------------------------------------------------
/*
 * returns z^((p-5)/8) = z^(2^252-3)
 * used to compute square roots since we have p=5 (mod 8); see Cohen and Frey.
 */
CUDA_CALLABLE
void pow22523(f51t::element& out, const f51t::element& z) noexcept;
} // namespace sxt::f51o
