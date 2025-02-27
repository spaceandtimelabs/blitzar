#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/fieldgk/operation/add.h"
#include "sxt/fieldgk/operation/mul.h"

namespace sxt::fgko {
//--------------------------------------------------------------------------------------------------
// muladd
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void muladd(fgkt::element& s, const fgkt::element& a, const fgkt::element& b,
            const fgkt::element& c) noexcept {
  mul(s, a, b);
  add(s, s, c);
}
} // namespace sxt::fgko
