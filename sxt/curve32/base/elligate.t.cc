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
#include "sxt/curve32/base/elligate.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/type/element.h"

using namespace sxt;
using namespace sxt::c32b;

TEST_CASE("we can elligate points") {
  f32t::element r = {0x3ab665f, 0x1cab33d, 0x2ff30ae, 0x1ea4bc0, 0xe069b0,
                     0x79cf12,  0x1e76416, 0x137e102, 0x2db995e, 0x3df46a};
  f32t::element x, y;
  int notsquare;
  apply_elligator(x, y, &notsquare, r);

  REQUIRE(notsquare == 0);

  REQUIRE(x[0] == 0x28d0ac);
  REQUIRE(x[1] == 0x47263);
  REQUIRE(x[2] == 0x704d63);
  REQUIRE(x[3] == 0x5cd4ec);
  REQUIRE(x[4] == 0x11cf896);
  REQUIRE(x[5] == 0x138496a);
  REQUIRE(x[6] == 0x652590);
  REQUIRE(x[7] == 0x9bdf8c);
  REQUIRE(x[8] == 0x3500e2);
  REQUIRE(x[9] == 0x54d916);

  REQUIRE(y[0] == 0x36de3fe);
  REQUIRE(y[1] == 0x1ff4134);
  REQUIRE(y[2] == 0x1f432af);
  REQUIRE(y[3] == 0x1b3dfe8);
  REQUIRE(y[4] == 0x1b37506);
  REQUIRE(y[5] == 0xb2f56e);
  REQUIRE(y[6] == 0xb3ab0a);
  REQUIRE(y[7] == 0x12fb237);
  REQUIRE(y[8] == 0x26c8f99);
  REQUIRE(y[9] == 0x1b0bf99);
}
