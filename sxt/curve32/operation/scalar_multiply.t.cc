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
#include "sxt/curve32/operation/scalar_multiply.h"

#include <iostream>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve32/operation/add.h"
#include "sxt/curve32/type/element_p3.h"

using namespace sxt;
using namespace sxt::c32o;

TEST_CASE("we can multiply elements by a scalar") {
  c32t::element_p3 g{
      {0x325d51a, 0x18b5823, 0xf6592a, 0x104a92d, 0x1a4b31d, 0x1d6dc5c, 0x27118fe, 0x7fd814,
       0x13cd6e5, 0x85a4db},
      {0x2666658, 0x1999999, 0xcccccc, 0x1333333, 0x1999999, 0x666666, 0x3333333, 0xcccccc,
       0x2666666, 0x1999999},
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0x1b7dda3, 0x1a2ace9, 0x25eadbb, 0x3ba8a, 0x83c27e, 0xabe37d, 0x1274732, 0xccacdd, 0xfd78b7,
       0x19e1d7c},
  };

  c32t::element_p3 res;

  SECTION("verify multiply by 1") {
    unsigned char a[32] = {1};
    scalar_multiply255(res, a, g);
    REQUIRE(res.X == f32t::element{0x74da0, 0x1c3f260, 0x1f15f57, 0xafd3a6, 0x18ea71b, 0xd0fedd,
                                   0x1344783, 0xcaacf6, 0xd6a292, 0x1ce0e84});
    REQUIRE(res.Y == f32t::element{0x309111b, 0x11323b9, 0x105b38b, 0x1a33284, 0xcb1867, 0x33cffa,
                                   0xccbfea, 0x1c8c4f3, 0xc9cbde, 0x1d9e6fe});
    REQUIRE(res.Z == f32t::element{0xcb5570, 0xd7eca8, 0x147206e, 0x18bff25, 0x2fdde81, 0x140c3f8,
                                   0x3ffefe4, 0x13af62f, 0x2fc3ed6, 0xd060bd});
    REQUIRE(res.T == f32t::element{0x5d7b7, 0x365b80, 0xc11913, 0x1bfdc85, 0x13eec15, 0x40cbe4,
                                   0x1c36c69, 0x1d5572b, 0xabb541, 0x10b3ed0});
  }

  SECTION("verify multiply by 2") {
    unsigned char a[32] = {2};
    scalar_multiply255(res, a, g);
    REQUIRE(res.X == f32t::element{0x98c2a5, 0xe3d5db, 0xe2c33d, 0x5e211f, 0x39dc2e5, 0xfbeda9,
                                   0x1dfd683, 0x182f2f2, 0x99d530, 0x1f9b7});
    REQUIRE(res.Y == f32t::element{0xcc710d, 0x1bdf020, 0x3104888, 0xd5d67e, 0x143f8e9, 0x3b0f5a,
                                   0x1e35570, 0xd2355b, 0x325b548, 0x1ed4d36});
    REQUIRE(res.Z == f32t::element{0x2b71779, 0x96f0c2, 0x1dd8f9c, 0x128015, 0x177f9da, 0x16757d2,
                                   0x337a180, 0x1da8c95, 0x3870bb8, 0x497c62});
    REQUIRE(res.T == f32t::element{0x37533aa, 0x11f341a, 0x1e7b884, 0x1efd1e8, 0x3f5e6a4, 0x95c253,
                                   0x2938e38, 0x12a89c2, 0x37b2d9e, 0x79537e});
  }

  SECTION("verify we can multiply by an exponent with a[31] > 127") {
    uint8_t a1[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125,
    };
    uint8_t a2[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125,
    };
    uint8_t a3[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 250,
    };
    REQUIRE(a3[31] > 127);
    scalar_multiply255(res, a1, g);
    auto expected_res = res;

    scalar_multiply255(res, a2, g);
    c32o::add(expected_res, expected_res, res);

    scalar_multiply(res, basct::span<uint8_t>{a3, 32}, g);
    REQUIRE(res == expected_res);
  }
}
