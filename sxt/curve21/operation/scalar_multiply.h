/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#pragma once

#include <cstdint>
#include <type_traits>

#include "sxt/base/container/span.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::s25t {
struct element;
}

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// scalar_multiply255
//--------------------------------------------------------------------------------------------------
/*
 h = a * p
 where a = a[0]+256*a[1]+...+256^31 a[31]

 Preconditions:
 a[31] <= 127

 p is public
 */
CUDA_CALLABLE
void scalar_multiply255(c21t::element_p3& h, const unsigned char* a,
                        const c21t::element_p3& p) noexcept;

//--------------------------------------------------------------------------------------------------
// scalar_multiply
//--------------------------------------------------------------------------------------------------
/*
 h = a * p
 where a = a[0]+256*a[1]+...+256^31 a[31]
 */
CUDA_CALLABLE
void scalar_multiply(c21t::element_p3& h, basct::cspan<uint8_t> a,
                     const c21t::element_p3& p) noexcept;

template <class T, std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T>>* = nullptr>
void scalar_multiply(c21t::element_p3& h, T a, const c21t::element_p3& p) noexcept {
  scalar_multiply(h, basct::cspan<uint8_t>{reinterpret_cast<uint8_t*>(&a), sizeof(a)}, p);
}

void scalar_multiply(c21t::element_p3& h, const s25t::element& a,
                     const c21t::element_p3& p) noexcept;
} // namespace sxt::c21o
