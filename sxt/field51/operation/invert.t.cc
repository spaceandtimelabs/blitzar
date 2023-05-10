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
#include "sxt/field51/operation/invert.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/type/element.h"
#include "sxt/field51/type/literal.h"

using namespace sxt;
using namespace sxt::f51o;
using namespace sxt::f51t;

TEST_CASE("we can invert field elements") {
  auto e = 0x123_f51;
  f51t::element ei;
  invert(ei, e);
  f51t::element res;
  mul(res, e, ei);
  REQUIRE(res == 0x1_f51);
}
