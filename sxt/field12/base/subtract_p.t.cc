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
#include "sxt/field12/base/subtract_p.h"

#include <random>

#include "sxt/base/test/unit_test.h"
#include "sxt/field12/base/constants.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::f12b;

// __global__ void subtract_p_device(uint64_t* __restrict__ ret, const uint64_t* __restrict__ a) {
//   subtract_p(ret, a);
// }

TEST_CASE("subtract_p (subtraction with the modulus) can handle computation") {
  SECTION("with minimum value") {
    constexpr std::array<uint64_t, 6> a = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    std::array<uint64_t, 6> ret;

    subtract_p(ret.data(), a.data());

    REQUIRE(ret == a);
  }

  SECTION("with a value below the modulus") {
    constexpr std::array<uint64_t, 6> a = {0xb9feffffffffaaaa, 0x1eabfffeb153ffff,
                                           0x6730d2a0f6b0f624, 0x64774b84f38512bf,
                                           0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
    std::array<uint64_t, 6> ret;

    subtract_p(ret.data(), a.data());

    REQUIRE(ret == a);
  }

  SECTION("with a value equal to the modulus") {
    constexpr std::array<uint64_t, 6> expect = {0, 0, 0, 0, 0, 0};
    std::array<uint64_t, 6> ret;

    subtract_p(ret.data(), p_v.data());

    REQUIRE(expect == ret);
  }

  SECTION("with a value above the modulus") {
    constexpr std::array<uint64_t, 6> a = {0xb9feffffffffaaae, 0x1eabfffeb153ffff,
                                           0x6730d2a0f6b0f624, 0x64774b84f38512bf,
                                           0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
    constexpr std::array<uint64_t, 6> expect = {0x3, 0x0, 0x0, 0x0, 0x0, 0x0};
    std::array<uint64_t, 6> ret;

    subtract_p(ret.data(), a.data());

    REQUIRE(expect == ret);
  }

  SECTION("with maximum value") {
    constexpr std::array<uint64_t, 6> a = {0xffffffffffffffff, 0xffffffffffffffff,
                                           0xffffffffffffffff, 0xffffffffffffffff,
                                           0xffffffffffffffff, 0xffffffffffffffff};
    constexpr std::array<uint64_t, 6> expect = {0x4601000000005554, 0xe15400014eac0000,
                                                0x98cf2d5f094f09db, 0x9b88b47b0c7aed40,
                                                0xb4e45849bcb45328, 0xe5feee15c6801965};
    std::array<uint64_t, 6> ret;

    subtract_p(ret.data(), a.data());

    REQUIRE(expect == ret);
  }
}

// TEST_CASE("subtract_p (subtraction with the modulus) can handle computation") {
//   SECTION("matches non GPU version") {
//     memmg::managed_array<std::array<uint64_t, 6>> a(1, memr::get_managed_device_resource());
//     memmg::managed_array<std::array<uint64_t, 6>> ret(1, memr::get_managed_device_resource());

//     std::random_device rd;
//     std::mt19937_64 gen(rd());
//     std::uniform_int_distribution<uint64_t> uni_dis;

//     constexpr unsigned iter = 100;
//     for (unsigned i = 0; i < iter; ++i) {
//       for (unsigned i = 0; i < 6; ++i) {
//         a[0][i] = uni_dis(gen);
//       }

//       std::array<uint64_t, 6> ret_expected;
//       subtract_p(ret_expected.data(), a[0].data());

//       subtract_p_device<<<1, 1>>>(ret.data(), a.data());
//       cudaDeviceSynchronize();

//       REQUIRE(ret[0] == ret_expected);
//     }
//   }
// }
