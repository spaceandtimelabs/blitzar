/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2023
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/field51/operation/invert.h"

#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/sq.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// invert
//--------------------------------------------------------------------------------------------------
void invert(f51t::element& out, const f51t::element& z) noexcept {
  f51t::element t0, t1, t2, t3;
  int i;

  sq(t0, z);
  sq(t1, t0);
  sq(t1, t1);
  mul(t1, z, t1);
  mul(t0, t0, t1);
  sq(t2, t0);
  mul(t1, t1, t2);
  sq(t2, t1);
  for (i = 1; i < 5; ++i) {
    sq(t2, t2);
  }
  mul(t1, t2, t1);
  sq(t2, t1);
  for (i = 1; i < 10; ++i) {
    sq(t2, t2);
  }
  mul(t2, t2, t1);
  sq(t3, t2);
  for (i = 1; i < 20; ++i) {
    sq(t3, t3);
  }
  mul(t2, t3, t2);
  for (i = 1; i < 11; ++i) {
    sq(t2, t2);
  }
  mul(t1, t2, t1);
  sq(t2, t1);
  for (i = 1; i < 50; ++i) {
    sq(t2, t2);
  }
  mul(t2, t2, t1);
  sq(t3, t2);
  for (i = 1; i < 100; ++i) {
    sq(t3, t3);
  }
  mul(t2, t3, t2);
  for (i = 1; i < 51; ++i) {
    sq(t2, t2);
  }
  mul(t1, t2, t1);
  for (i = 1; i < 6; ++i) {
    sq(t1, t1);
  }
  mul(out, t1, t0);
}
} // namespace sxt::f51o
