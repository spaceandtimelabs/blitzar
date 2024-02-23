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
#include "sxt/field32/type/element.h"

#include <sstream>

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/base/byte_conversion.h"
#include "sxt/field32/type/literal.h"
#include "sxt/field51/base/byte_conversion.h"
#include "sxt/field51/type/element.h"

using namespace sxt;
using namespace sxt::f32t;
using namespace sxt::f51t;

TEST_CASE("element conversion") {
  std::ostringstream oss;
  SECTION("of zero prints as zero") {
    f32t::element e{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    oss << e;
    REQUIRE(oss.str() == "0x0_f32");
  }

  SECTION("of one prints as one") {
    f32t::element e{1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    oss << e;
    REQUIRE(oss.str() == "0x1_f32");
  }

  SECTION("of Edwards D prints as expected") {
    f32t::element e{56195235, 13857412, 51736253, 6949390,  114729,
                    24766616, 60832955, 30306712, 48412415, 21499315};
    oss << e;
    REQUIRE(oss.str() == "0x52036cee2b6ffe738cc740797779e89800700a4d4141d8ab75eb4dca135978a3_f32");
  }

  SECTION("of ed25519_sqrtam2 prints as expected") {
    f32t::element e{0x3457e06, 0x1812abf, 0x350598d, 0x8a5be8,  0x316874f,
                    0x1fc4f7e, 0x1846e01, 0xd77a4f,  0x3460a00, 0x3c9bb7};
    oss << e;
    REQUIRE(oss.str() == "0xf26edf460a006bbd27b08dc03fc4f7ec5a1d3d14b7d1a82cc6e04aaff457e06_f32");
  }
}

TEST_CASE("element equality") {
  f32t::element e{1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  REQUIRE(e == e);
}

TEST_CASE("matches field51 - WILL NOT BE CHECKED IN") {
  SECTION("ed25519_sqrtam2") {
    f32t::element e{0x3457e06, 0x1812abf, 0x350598d, 0x8a5be8,  0x316874f,
                    0x1fc4f7e, 0x1846e01, 0xd77a4f,  0x3460a00, 0x3c9bb7};
    f51t::element e51{1693982333959686, 608509411481997, 2235573344831311, 947681270984193,
                      266558006233600};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element X") {
    f32t::element e = {0x325d51a, 0x18b5823, 0xf6592a, 0x104a92d, 0x1a4b31d,
                       0x1d6dc5c, 0x27118fe, 0x7fd814, 0x13cd6e5, 0x85a4db};
    f51t::element e51 = {3990542415680775, 3398198340507945, 4322667446711068, 2814063955482877,
                         2839572215813860};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element Y") {
    f32t::element e = {0x2666658, 0x1999999, 0xcccccc, 0x1333333, 0x1999999,
                       0x666666,  0x3333333, 0xcccccc, 0x2666666, 0x1999999};
    f51t::element e51 = {1801439850948184, 1351079888211148, 450359962737049, 900719925474099,
                         1801439850948198};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element Z") {
    f32t::element e = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    f51t::element e51 = {1, 0, 0, 0, 0};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element T") {
    f32t::element e = {0x1b7dda3, 0x1a2ace9, 0x25eadbb, 0x3ba8a,  0x83c27e,
                       0xabe37d,  0x1274732, 0xccacdd,  0xfd78b7, 0x19e1d7c};
    f51t::element e51 = {1841354044333475, 16398895984059, 755974180946558, 900171276175154,
                         1821297809914039};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element curve scalar multiply test verify multiply by 1 - X") {
    f32t::element e = {0x74da0,  0x1c3f260, 0x1f15f57, 0xafd3a6, 0x18ea71b,
                       0xd0fedd, 0x1344783, 0xcaacf6,  0xd6a292, 0x1ce0e84};
    f51t::element e51 = {1987682947780000, 773294264508247, 919172218267419, 891376861726595,
                         2032146878734994};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element curve scalar multiply test verify multiply by 1 - Y") {
    f32t::element e = {0x309111b, 0x11323b9, 0x105b38b, 0x1a33284, 0xcb1867,
                       0x33cffa,  0xccbfea,  0x1c8c4f3, 0xc9cbde,  0x1d9e6fe};
    f51t::element e51 = {1210076552040731, 1843649357132683, 227873395513447, 2008892784295914,
                         2084244428540894};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element curve scalar multiply test verify multiply by 1 - Z") {
    f32t::element e = {0xcb5570,  0xd7eca8,  0x147206e, 0x18bff25, 0x2fdde81,
                       0x140c3f8, 0x3ffefe4, 0x13af62f, 0x2fc3ed6, 0xd060bd};
    f51t::element e51 = {949645736629616, 1741611742994542, 1410741651234433, 1385216073527268,
                         916455675412182};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element curve scalar multiply test verify multiply by 1 - T") {
    f32t::element e = {0x5d7b7,  0x365b80,  0xc11913,  0x1bfdc85, 0x13eec15,
                       0x40cbe4, 0x1c36c69, 0x1d5572b, 0xabb541,  0x10b3ed0};
    f51t::element e51 = {239066470012855, 1969715299817747, 284977811876885, 2064181377592425,
                         1175357540250945};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element curve scalar multiply test verify multiply by 2 - X") {
    f32t::element e = {0x98c2a5, 0xe3d5db,  0xe2c33d,  0x5e211f, 0x39dc2e5,
                       0xfbeda9, 0x1dfd683, 0x182f2f2, 0x99d530, 0x1f9b7};
    f51t::element e51 = {1002030577009317, 413985402962749, 1107992705352421, 1701819753420419,
                         8688124941616};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element curve scalar multiply test verify multiply by 2 - Y") {
    f32t::element e = {0xcc710d, 0x1bdf020, 0x3104888, 0xd5d67e,  0x143f8e9,
                       0x3b0f5a, 0x1e35570, 0xd2355b,  0x325b548, 0x1ed4d36};
    f51t::element e51 = {1961256026927373, 940468905986184, 259748503222505, 924506438980976,
                         2169563456582984};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element curve scalar multiply test verify multiply by 2 - Z") {
    f32t::element e = {0x2b71779, 0x96f0c2,  0x1dd8f9c, 0x128015,  0x177f9da,
                       0x16757d2, 0x337a180, 0x1da8c95, 0x3870bb8, 0x497c62};
    f51t::element e51 = {663843209942905, 81365301039004, 1580407463606746, 2087089281147264,
                         323194334940088};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element curve scalar multiply test verify multiply by 2 - T") {
    f32t::element e = {0x37533aa, 0x11f341a, 0x1e7b884, 0x1efd1e8, 0x3f5e6a4,
                       0x95c253,  0x2938e38, 0x12a89c2, 0x37b2d9e, 0x79537e};
    f51t::element e51 = {1263134504727466, 2180639216875652, 658647461258916, 1312984564731448,
                         533598071106974};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element curve we can elligate points element") {
    f32t::element e = {0x3ab665f, 0x1cab33d, 0x2ff30ae, 0x1ea4bc0, 0xe069b0,
                       0x79cf12,  0x1e76416, 0x137e102, 0x2db995e, 0x3df46a};
    f51t::element e51 = {2017384653874783ull, 2156344215810222ull, 535721083431344ull,
                         1371658101679126ull, 272479886743902ull};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element curve we can elligate points element result X") {
    f32t::element e = {0x28d0ac,  0x47263,  0x704d63, 0x5cd4ec, 0x11cf896,
                       0x138496a, 0x652590, 0x9bdf8c, 0x3500e2, 0x54d916};
    f51t::element e51 = {6774956778639475ull, 7163677697396064ull, 8128851215186067ull,
                         7440937162974605ull, 7128564859470047ull};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }

  SECTION("test element curve we can elligate points element result Y") {
    f32t::element e = {0x36de3fe, 0x1ff4134, 0x1f432af, 0x1b3dfe8, 0x1b37506,
                       0xb2f56e,  0xb3ab0a,  0x12fb237, 0x26c8f99, 0x1b0bf99};
    f51t::element e51 = {2248522005865470ull, 1916996945195695ull, 787068757439750ull,
                         1335669812341514ull, 1903247756136345ull};

    u_int8_t e_s[32];
    u_int8_t e51_s[32];

    f32b::to_bytes(e_s, e.data());
    f51b::to_bytes(e51_s, e51.data());

    REQUIRE(std::memcmp(e_s, e51_s, 32) == 0);
  }
}

/*
#include <iostream>

int main() {
  // To run tests, add f51t::element to the field51/type/element test and get the hex output.
  // Replace the hex output below in the "Element to convert" section.
  // Then add a test.

  // Element to convert.
  auto element = 0x6c2fe666c8f9997d91b9675614b2f56e6cdd41b67bfd0fa1957ffd04d36de3fe_f32;

  // Print as hex
  for (int i = 0; i < 10; ++i) {
    std::cout << "0x" << std::hex << (int)element[i] << ", ";
  }
  std::cout << std::endl;

  // Convert to bytes
  uint8_t s[32];
  f32b::to_bytes(s, element.data());

  // Convert bytes to element
  uint32_t h[10];
  f32b::from_bytes(h, s);

  // Verify conversion matches original element
  bool match = true;
  for (int i = 0; i < 10; ++i) {
    if (h[i] != element[i]) {
      match = false;
      std::cout << "Conversion failed at index " << i << "\n";
      for (int i = 0; i < 10; ++i) {
        std::cout << "0x" << std::hex << h[i] << " == 0x" << element[i] << "\n";
      }
    }
  }
  if (match) {
    std::cout << "Conversion matches original element\n";
  }

  return 0;
}
*/
