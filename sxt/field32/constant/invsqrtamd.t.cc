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
#include "sxt/field32/constant/invsqrtamd.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/type/literal.h"

using namespace sxt::f32cn;
using namespace sxt::f32t;

TEST_CASE("invsqrtamd") {
  REQUIRE(invsqrtamd == 0x786c8905cfaffca216c27b91fe01d8409d2f16175a4172be99c8fdaa805d40ea_f32);
}
