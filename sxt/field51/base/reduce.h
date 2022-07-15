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
#include "sxt/base/type/int.h"

namespace sxt::f51b {
//--------------------------------------------------------------------------------------------------
// reduce
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void reduce(uint64_t h[5], const uint64_t f[5]) noexcept;
} // namespace sxt::f51b
