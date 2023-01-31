/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/scalar25/operation/add.h"

#include "sxt/scalar25/operation/reduce.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// add_impl
//--------------------------------------------------------------------------------------------------
//
// Modified from libsodium's sodium_add to use only 33 bytes instead of 64
template <class T> CUDA_CALLABLE void add_impl(s25t::element& z, T& x, T& y) noexcept {
  auto z_data = z.data();
  auto x_data = x.data();
  auto y_data = y.data();
  uint_fast16_t carry = 0U;

  // x + y can produce at maximum a z = x + y with 33 bytes.
  // we iterate only through 32 bytes because we know
  // that the 33rd byte will be stored in the carry variable.
  for (size_t i = 0U; i < 32; i++) {
    carry += static_cast<uint_fast16_t>(x_data[i]) + static_cast<uint_fast16_t>(y_data[i]);
    z_data[i] = static_cast<uint8_t>(carry);
    carry >>= 8;
  }

  s25o::reduce33(z, static_cast<uint8_t>(carry));
}

//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void add(s25t::element& z, const s25t::element& x, const s25t::element& y) noexcept {
  add_impl(z, x, y);
}

CUDA_CALLABLE
void add(s25t::element& z, const volatile s25t::element& x,
         const volatile s25t::element& y) noexcept {
  add_impl(z, x, y);
}
} // namespace sxt::s25o
