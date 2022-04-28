/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t { struct element_cached; }

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// cmov
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void cmov(c21t::element_cached& t, const c21t::element_cached& u,
          unsigned char b) noexcept;

//--------------------------------------------------------------------------------------------------
// cmov8
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void cmov8(c21t::element_cached& t, const c21t::element_cached cached[8],
                         const signed char b) noexcept;
} // namespace sxt::c21o
