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
#include "sxt/field32/operation/notsquare.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/type/literal.h"

using namespace sxt;
using namespace sxt::f32o;
using sxt::f32t::operator""_f32;

TEST_CASE("we can detect if an element is not a square") {
  REQUIRE(notsquare(0x4_f32) == 0);
  REQUIRE(notsquare(0x123_f32) == 1);
  REQUIRE(notsquare(0x48674afb484b050fdcccf508dfb8ce91c364ab4d15584711cba01736e1c59deb_f32) == 1);
}
