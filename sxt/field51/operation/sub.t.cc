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
#include "sxt/field51/operation/sub.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/type/element.h"
#include "sxt/field51/type/literal.h"

using namespace sxt;
using namespace sxt::f51o;
using namespace sxt::f51t;

TEST_CASE("sub") {
  auto a = 0x5187889dbf41151c5858a2cc5924e0a9e34592749088b9866d452769294eca83_f51;
  auto b = 0x7397e3032f73c74bb068e931363dc16ea16bb74d68e16d331ca2f087c70eef70_f51;
  auto expected = 0x5defa59a8fcd4dd0a7efb99b22e71f3b41d9db2727a74c5350a236e1623fdb00_f51;

  f51t::element diff;

  sub(diff, a, b);

  REQUIRE(diff == expected);
}
