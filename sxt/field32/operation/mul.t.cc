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
#include "sxt/field32/operation/mul.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/type/element.h"
#include "sxt/field32/type/literal.h"

using namespace sxt;
using namespace sxt::f32o;
using namespace sxt::f32t;

TEST_CASE("mul") {
  SECTION("a * b = c") {
    f32t::element a{12775011, 23676731, 59192183, 16867033, 23768340,
                    1913951,  6461676,  11632506, 6764333,  17261081};
    f32t::element b{58368107, 31446661, 49405094, 30328970, 25558377,
                    26518888, 65879413, 25699118, 56941748, 10839393};
    f32t::element expected{10315675, 28647227, 66416440, 17953483, 38513412,
                           3169955,  47930313, 11586915, 1363003,  12222828};

    f32t::element ret;
    mul(ret, a, b);
    REQUIRE(ret == expected);
  }
}

TEST_CASE("we can multiply finite field elements") {
  auto e1 = 0x711a90c454965634b0962b2b4479551d887ad8d7f33d62f626648de22323dba0_f32;
  auto e2 = 0x48674afb484b050fdcccf508dfb8ce91c364ab4d15584711cba01736e1c59deb_f32;
  element res;

  SECTION("verify against precomputed values") {
    mul(res, e1, e2);
    auto expected_res = 0x6b9ecf8d8ab80bb3a98db6783ec540c9d2fab4684e954e733461f8f187a84ff8_f32;
    REQUIRE(res == expected_res);
  }

  SECTION("verify against one") {
    auto one = 0x1_f32;
    mul(res, one, one);
    REQUIRE(res == one);
  }

  SECTION("another one") {
    auto e = 0x123_f32;
    auto inv = 0x5e9208cc18a1de9208cc18a1de9208cc18a1de9208cc18a1de9208cc18a1de84_f32;
    f32t::element ei;
    mul(res, e, inv);

    // p_v + 1
    // Same return as in the curve25519 project's u32 mul implementation
    f32t::element ret_from_curve25519{67108846, 33554431, 67108863, 33554431, 67108863,
                                      33554431, 67108863, 33554431, 67108863, 33554431};

    REQUIRE(res == ret_from_curve25519);
  }

  SECTION("we can multiply by 32-bit integers") {
    mul32(res, e1, 0x123u);
    element expected;
    mul(expected, e1, 0x123_f32);
    REQUIRE(res == expected);
  }

  SECTION("1000 multiplies") {
    element ret;
    for (size_t i = 0; i < 1000; ++i) {
      mul(ret, e1, e1);
    }
    REQUIRE(ret == 0x2ad36c780fc54d4d11c2d325469d7571606c44ad4013dcfe4392e6731b07079_f32);
  }
}
