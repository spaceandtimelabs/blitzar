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
#include "sxt/scalar25/operation/product_mapper.h"

#include <cstddef>

#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::s25o;
using sxt::s25t::operator""_s25;

TEST_CASE("we can map two arrays of scalars to their product") {
  SECTION("map_index gives us the product of two scalars") {
    memmg::managed_array<s25t::element> a = {0x123_s25};
    memmg::managed_array<s25t::element> b = {0x321_s25};
    product_mapper mapper{a.data(), b.data()};
    REQUIRE(mapper.map_index(0) == a[0] * b[0]);
    s25t::element res;
    mapper.map_index(res, 0);
    REQUIRE(res == a[0] * b[0]);
  }
}
