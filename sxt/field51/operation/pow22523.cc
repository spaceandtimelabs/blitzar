/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/field51/operation/pow22523.h"

#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/square.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// pow22523
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void pow22523(f51t::element& out, const f51t::element& z) noexcept {
  f51t::element t0, t1, t2;
  int i;

  square(t0, z);
  square(t1, t0);
  square(t1, t1);
  mul(t1, z, t1);
  mul(t0, t0, t1);
  square(t0, t0);
  mul(t0, t1, t0);
  square(t1, t0);
  for (i = 1; i < 5; ++i) {
    square(t1, t1);
  }
  mul(t0, t1, t0);
  square(t1, t0);
  for (i = 1; i < 10; ++i) {
    square(t1, t1);
  }
  mul(t1, t1, t0);
  square(t2, t1);
  for (i = 1; i < 20; ++i) {
    square(t2, t2);
  }
  mul(t1, t2, t1);
  for (i = 1; i < 11; ++i) {
    square(t1, t1);
  }
  mul(t0, t1, t0);
  square(t1, t0);
  for (i = 1; i < 50; ++i) {
    square(t1, t1);
  }
  mul(t1, t1, t0);
  square(t2, t1);
  for (i = 1; i < 100; ++i) {
    square(t2, t2);
  }
  mul(t1, t2, t1);
  for (i = 1; i < 51; ++i) {
    square(t1, t1);
  }
  mul(t0, t1, t0);
  square(t0, t0);
  square(t0, t0);
  mul(out, t0, z);
}
} // namespace sxt::f51o
