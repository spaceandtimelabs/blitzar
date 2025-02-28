#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/fieldgk/operation/add.h"
#include "sxt/fieldgk/operation/mul.h"

namespace sxt::fgko {
//--------------------------------------------------------------------------------------------------
// muladd
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void muladd(fgkt::element& s, const fgkt::element& a, const fgkt::element& b,
                                 const fgkt::element& c) noexcept {
  auto cp = c;
  mul(s, a, b);
  add(s, s, cp);
}
} // namespace sxt::fgko
