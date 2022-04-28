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

namespace sxt::c21t { struct element_p3; }

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// scalar_multiply
//--------------------------------------------------------------------------------------------------
/*
 h = a * p
 where a = a[0]+256*a[1]+...+256^31 a[31]

 Preconditions:
 a[31] <= 127

 p is public
 */
CUDA_CALLABLE
void scalar_multiply(c21t::element_p3& h, const unsigned char* a,
                     const c21t::element_p3& p) noexcept;
}  // namespace sxt::c21o
