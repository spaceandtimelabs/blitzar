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
#include "sxt/field25/operation/add.h"

#include "sxt/base/device/synchronization.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field25/constant/zero.h"
#include "sxt/field25/random/element.h"
#include "sxt/field25/type/element.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::f25o;

__global__ void add(f25t::element* __restrict__ ret, const f25t::element* __restrict__ a,
                    const f25t::element* __restrict__ b) {
  add(ret[0], a[0], b[0]);
}

TEST_CASE("addition") {
  SECTION("of a random field element and zero returns the random field element") {
    f25t::element a;
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(a, rng);

    f25t::element ret;

    add(ret, a, f25cn::zero_v);

    REQUIRE(a == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random bn254 base field elements generated using the SAGE library.
    constexpr f25t::element a{0x1149c21473b043fd, 0x5610b2a5c08c7ecf, 0xc9e31f2d914c45b5,
                              0x066031eb7a3ca7fd};
    constexpr f25t::element b{0x13f757e660d431b8, 0x8a86bc6a237b60d5, 0x6f91e11522e9b96d,
                              0x10ce4233724f624b};
    constexpr f25t::element expected{0x254119fad48475b5, 0xe0976f0fe407dfa4, 0x39750042b435ff22,
                                     0x172e741eec8c0a49};
    f25t::element ret;

    add(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("of a pre-computed value the modulus minus one returns expected value") {
    // Random bn254 base field element generated using the SAGE library.
    constexpr f25t::element a{0x1149c21473b043fd, 0x5610b2a5c08c7ecf, 0xc9e31f2d914c45b5,
                              0x066031eb7a3ca7fd};
    constexpr f25t::element b{0x3c208c16d87cfd46, 0x97816a916871ca8d, 0xb85045b68181585d,
                              0x30644e72e131a029};
    constexpr f25t::element expected{0x1149c21473b043fc, 0x5610b2a5c08c7ecf, 0xc9e31f2d914c45b5,
                                     0x066031eb7a3ca7fd};
    f25t::element ret;

    add(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("of the modulus minus one and one returns zero") {
    constexpr f25t::element a{0x3c208c16d87cfd46, 0x97816a916871ca8d, 0xb85045b68181585d,
                              0x30644e72e131a029};
    constexpr f25t::element b{1, 0, 0, 0};
    f25t::element ret;

    add(ret, a, b);

    REQUIRE(f25cn::zero_v == ret);
  }
}

TEST_CASE("addition on device") {
  memmg::managed_array<f25t::element> a(1, memr::get_managed_device_resource());
  memmg::managed_array<f25t::element> b(1, memr::get_managed_device_resource());
  memmg::managed_array<f25t::element> ret(1, memr::get_managed_device_resource());

  SECTION("of a random field element and zero returns the random field element") {
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(a[0], rng);
    b[0] = f25cn::zero_v;

    add<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(a[0] == ret[0]);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random bn254 base field elements generated using the SAGE library.
    a[0] = {0x1149c21473b043fd, 0x5610b2a5c08c7ecf, 0xc9e31f2d914c45b5, 0x066031eb7a3ca7fd};
    b[0] = {0x13f757e660d431b8, 0x8a86bc6a237b60d5, 0x6f91e11522e9b96d, 0x10ce4233724f624b};
    constexpr f25t::element expected{0x254119fad48475b5, 0xe0976f0fe407dfa4, 0x39750042b435ff22,
                                     0x172e741eec8c0a49};

    add<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(expected == ret[0]);
  }

  SECTION("of a pre-computed value the modulus minus one returns expected value") {
    // Random bn254 base field element generated using the SAGE library.
    a[0] = {0x1149c21473b043fd, 0x5610b2a5c08c7ecf, 0xc9e31f2d914c45b5, 0x066031eb7a3ca7fd};
    b[0] = {0x3c208c16d87cfd46, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029};
    constexpr f25t::element expected{0x1149c21473b043fc, 0x5610b2a5c08c7ecf, 0xc9e31f2d914c45b5,
                                     0x066031eb7a3ca7fd};

    add<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(expected == ret[0]);
  }

  SECTION("of the modulus minus one and one returns zero") {
    a[0] = {0x3c208c16d87cfd46, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029};
    b[0] = {1, 0, 0, 0};

    add<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(f25cn::zero_v == ret[0]);
  }

  SECTION("matches a random host output") {
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(a[0], rng);
    f25rn::generate_random_element(b[0], rng);
    const f25t::element a_host = a[0];
    const f25t::element b_host = b[0];
    f25t::element ret_host;

    add(ret_host, a_host, b_host);

    add<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(ret_host == ret[0]);
  }
}
