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
#include "sxt/field25/type/element.h"

#include <sstream>

#include "sxt/base/test/unit_test.h"
#include "sxt/field25/base/constants.h"

using namespace sxt;
using namespace sxt::f25t;

TEST_CASE("element conversion") {
  std::ostringstream oss;

  SECTION("of zero prints as zero") {
    element e{0, 0, 0, 0, 0, 0};
    oss << e;
    REQUIRE(oss.str() == "0x0_f12");
  }

  SECTION("of one in Montgomery form prints as one") {
    element e{0x760900000002fffd, 0xebf4000bc40c0002, 0x5f48985753c758ba,
              0x77ce585370525745, 0x5c071a97a256ec6d, 0x15f65ec3fa80e493};
    oss << e;
    REQUIRE(oss.str() == "0x1_f12");
  }

  SECTION("of the modulus prints as zero") {
    element e(f12b::p_v.data());
    oss << e;
    REQUIRE(oss.str() == "0x0_f12");
  }

  SECTION("of the modulus minus one prints a pre-computed value") {
    element e{0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624,
              0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
    oss << e;
    REQUIRE(oss.str() == "0x5024ae85084d9b05dbd438f06fc594c4cdfa0709adc84d632f22927e21b885b9ecaed89"
                         "d8bb0503c52b7da6c7f4628b_f12");
  }

  SECTION("of a pre-computed value returns a pre-computed value") {
    const uint64_t e_array[6] = {0, 1, 2, 3, 4, 5};
    element e{e_array};
    oss << e;
    REQUIRE(oss.str() == "0x55cb35bae0c5253c1f42d6d581fc0f94780c21ab9af0f38e8699a7f484af974311fee26"
                         "c2fec135aadf4b21a3346b64_f12");
  }
}
