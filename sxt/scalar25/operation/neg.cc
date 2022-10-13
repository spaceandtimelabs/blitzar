/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/scalar25/operation/neg.h"

#include "sxt/scalar25/base/reduce.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// neg
//--------------------------------------------------------------------------------------------------
//
// Modified from libsodium's crypto_core_ed25519_scalar_negate
CUDA_CALLABLE
void neg(s25t::element& n, const s25t::element& s) noexcept {
  uint8_t t_[64] = {/* 0 */
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    /* 2^252+27742317777372353535851937790883648493 */
                    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde,
                    0xf9, 0xde, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x10};

  uint_fast16_t c = 0U;
  auto s_data = s.data();
  for (size_t i = 0U; i < 32U; i++) {
    c = -static_cast<uint_fast16_t>(s_data[i]) - c;
    t_[i] = static_cast<uint8_t>(c);
    c = (c >> 8) & 1U;
  }

  for (size_t i = 32U; i < 64U; i++) {
    c = static_cast<uint_fast16_t>(t_[i]) - c;
    t_[i] = static_cast<uint8_t>(c);
    c = (c >> 8) & 1U;
  }

  s25b::reduce64(n, t_);
}
} // namespace sxt::s25o
