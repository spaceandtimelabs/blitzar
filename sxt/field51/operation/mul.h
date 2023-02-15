#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::f51t {
class element;
}

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// mul
//--------------------------------------------------------------------------------------------------
/*
 h = f * g
 Can overlap h with f or g.
 */
CUDA_CALLABLE
void mul(f51t::element& h, const f51t::element& f, const f51t::element& g) noexcept;

CUDA_CALLABLE
void mul(volatile f51t::element& h, const f51t::element& f, const f51t::element& g) noexcept;
} // namespace sxt::f51o
