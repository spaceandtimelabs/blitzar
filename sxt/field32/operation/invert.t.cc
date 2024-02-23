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
#include "sxt/field32/operation/invert.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/type/element.h"
#include "sxt/field32/type/literal.h"

using namespace sxt;
using namespace sxt::f32o;
using namespace sxt::f32t;

TEST_CASE("we can invert field elements") {
  auto e = 0x123_f32;
  f32t::element ei;
  invert(ei, e);
  REQUIRE(ei == 0x5e9208cc18a1de9208cc18a1de9208cc18a1de9208cc18a1de9208cc18a1de84_f32);
  f32t::element res;
  mul(res, e, ei);

  // p_v + 1
  // Same return as in the curve25519 project's u32 mul implementation
  f32t::element ret_from_curve25519{67108846, 33554431, 67108863, 33554431, 67108863,
                                    33554431, 67108863, 33554431, 67108863, 33554431};

  REQUIRE(res == ret_from_curve25519);
}
