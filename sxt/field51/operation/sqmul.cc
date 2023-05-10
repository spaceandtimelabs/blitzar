/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/field51/operation/sqmul.h"

#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/sq.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// sqmul
//--------------------------------------------------------------------------------------------------
void sqmul(f51t::element& s, int n, const f51t::element& a) noexcept {
  for (int i = 0; i < n; i++) {
    f51o::sq(s, s);
  }
  f51o::mul(s, s, a);
}
} // namespace sxt::f51o
