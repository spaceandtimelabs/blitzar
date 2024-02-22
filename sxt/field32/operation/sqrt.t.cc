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
#include "sxt/field32/operation/sqrt.h"

#include <sstream>

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/type/literal.h"

using namespace sxt;
using namespace sxt::f32o;
using f32t::operator""_f32;

TEST_CASE("we can compute square roots of field elements") {
  f32t::element rt;
  auto x = 0x4_f32;
  REQUIRE(sqrt(rt, x) == 0);
  f32t::element pow2;
  f32o::mul(pow2, rt, rt);

  std::ostringstream oss;
  oss << pow2;
  REQUIRE(oss.str() == "0x4_f32");

  f32t::element p_v_plus_4{67108849, 33554431, 67108863, 33554431, 67108863, 33554431, 67108863, 33554431, 67108863, 33554431};
  REQUIRE(pow2 == p_v_plus_4);
  
  REQUIRE(sqrt(rt, 0x123_f32) != 0);
}
