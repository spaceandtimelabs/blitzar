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
#include "sxt/curve_g1/operation/compression.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_g1/constant/generator.h"
#include "sxt/curve_g1/type/compressed_element.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/field12/constant/one.h"
#include "sxt/field12/constant/zero.h"
#include "sxt/field12/operation/mul.h"
#include "sxt/field12/type/element.h"

using namespace sxt;
using namespace sxt::cg1o;

TEST_CASE("compression correctly marks bits related to the") {
  constexpr uint8_t compressed_bit{static_cast<uint8_t>(1) << 7};
  constexpr uint8_t infinity_bit{static_cast<uint8_t>(1) << 6};
  constexpr uint8_t lexicographically_largest_bit{static_cast<uint8_t>(1) << 5};

  SECTION("generator") {
    cg1t::compressed_element result;

    compress(result, cg1cn::generator_p2_v);

    REQUIRE((result.data()[47] & compressed_bit) >> 7);
    REQUIRE(!((result.data()[47] & infinity_bit) >> 6));
    REQUIRE(!((result.data()[47] & lexicographically_largest_bit) >> 5));
  }

  SECTION("point at infinity") {
    cg1t::compressed_element result;

    cg1t::element_p2 inf_elm{f12cn::zero_v, f12cn::one_v, f12cn::zero_v};

    compress(result, inf_elm);

    REQUIRE((result.data()[47] & compressed_bit) >> 7);
    REQUIRE((result.data()[47] & infinity_bit) >> 6);
    REQUIRE(!((result.data()[47] & lexicographically_largest_bit) >> 5));
  }

  SECTION("lexicographically largest element") {
    cg1t::compressed_element result;

    // This element is not on the curve. It is just used for testing compression.
    f12t::element e{0x43f5fffffffcaaae, 0x32b7fff2ed47fffd, 0x07e83a49a2e99d69,
                    0xeca8f3318332bb7a, 0xef148d1ea0f4c069, 0x040ab3263eff0206};
    cg1t::element_p2 ll_elm{f12cn::one_v, e, f12cn::one_v};

    compress(result, ll_elm);

    REQUIRE((result.data()[47] & compressed_bit) >> 7);
    REQUIRE(!((result.data()[47] & infinity_bit) >> 6));
    REQUIRE(((result.data()[47] & lexicographically_largest_bit) >> 5));

    // Make point at infinity.
    ll_elm.Z = f12cn::zero_v;

    compress(result, ll_elm);

    REQUIRE((result.data()[47] & compressed_bit) >> 7);
    REQUIRE((result.data()[47] & infinity_bit) >> 6);
    REQUIRE(!((result.data()[47] & lexicographically_largest_bit) >> 5));
  }
}

TEST_CASE("batch compression correctly marks bits related to the") {
  constexpr uint8_t compressed_bit{static_cast<uint8_t>(1) << 7};
  constexpr uint8_t infinity_bit{static_cast<uint8_t>(1) << 6};
  constexpr uint8_t lexicographically_largest_bit{static_cast<uint8_t>(1) << 5};

  SECTION("generator") {
    std::vector<cg1t::compressed_element> res_vec{cg1t::compressed_element{},
                                                  cg1t::compressed_element{}};
    basct::span<cg1t::compressed_element> results{res_vec};
    const std::vector<cg1t::element_p2> gen_vec{cg1cn::generator_p2_v, cg1cn::generator_p2_v};
    basct::cspan<cg1t::element_p2> generators{gen_vec};

    batch_compress(results, generators);

    for (size_t i = 0; i < generators.size(); ++i) {
      REQUIRE((results[i].data()[47] & compressed_bit) >> 7);
      REQUIRE(!((results[i].data()[47] & infinity_bit) >> 6));
      REQUIRE(!((results[i].data()[47] & lexicographically_largest_bit) >> 5));
    }
  }
}

TEST_CASE("compression of two G1 curve elements that are") {
  SECTION("different remain different") {
    cg1t::compressed_element ce1;
    cg1t::compressed_element ce2;

    compress(ce1, cg1cn::generator_p2_v);
    compress(ce2, cg1t::element_p2::identity());

    REQUIRE(ce1 != ce2);
  }

  SECTION("projected remain the same") {
    cg1t::compressed_element ce1;
    cg1t::compressed_element ce2;

    constexpr f12t::element z{0xba7afa1f9a6fe250, 0xfa0f5b595eafe731, 0x3bdc477694c306e7,
                              0x2149be4b3949fa24, 0x64aa6e0649b2078c, 0x12b108ac33643c3e};
    f12t::element gpx_z;
    f12t::element gpy_z;
    f12o::mul(gpx_z, cg1cn::generator_p2_v.X, z);
    f12o::mul(gpy_z, cg1cn::generator_p2_v.Y, z);
    cg1t::element_p2 projected_generator{gpx_z, gpy_z, z};

    compress(ce1, cg1cn::generator_p2_v);
    compress(ce2, projected_generator);

    REQUIRE(ce1 == ce2);
  }
}
