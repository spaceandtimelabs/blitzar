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
/**
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#include "sxt/field12/operation/mul.h"

#include "sxt/base/device/synchronization.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field12/constant/zero.h"
#include "sxt/field12/random/element.h"
#include "sxt/field12/type/element.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::f12o;

__global__ void mul(f12t::element* __restrict__ ret, const f12t::element* __restrict__ a,
                    const f12t::element* __restrict__ b) {
  mul(ret[0], a[0], b[0]);
}

TEST_CASE("multiplication") {
  SECTION("of a pre-computed value and zero returns zero") {
    constexpr f12t::element a{0x14fd599a70a077d8, 0xf10f187e64b9426d, 0x8ee7d30c9b4bab84,
                              0x1bcd1f840d3ed0f0, 0xcd14c6f310abdc3d, 0x69d558ed64d6a25f};
    f12t::element ret;

    mul(ret, a, f12cn::zero_v);

    REQUIRE(ret == f12cn::zero_v);
  }

  SECTION("of pre-computed values returns expected value") {
    constexpr f12t::element a{0x0397a38320170cd4, 0x734c1b2c9e761d30, 0x5ed255ad9a48beb5,
                              0x095a3c6b22a7fcfc, 0x2294ce75d4e26a27, 0x13338bd870011ebb};
    constexpr f12t::element b{0xb9c3c7c5b1196af7, 0x2580e2086ce335c1, 0xf49aed3d8a57ef42,
                              0x41f281e49846e878, 0xe0762346c38452ce, 0x0652e89326e57dc0};
    constexpr f12t::element expected{0xf96ef3d711ab5355, 0xe8d459ea00f148dd, 0x53f7354a5f00fa78,
                                     0x9e34a4f3125c5f83, 0x3fbe0c47ca74c19e, 0x01b06a8bbd4adfe4};
    f12t::element ret;

    mul(ret, a, b);

    REQUIRE(expected == ret);
  }
}

TEST_CASE("multiplication on device") {
  memmg::managed_array<f12t::element> a(1, memr::get_managed_device_resource());
  memmg::managed_array<f12t::element> b(1, memr::get_managed_device_resource());
  memmg::managed_array<f12t::element> ret(1, memr::get_managed_device_resource());

  SECTION("of a pre-computed value and zero returns zero") {
    a[0] = {0x14fd599a70a077d8, 0xf10f187e64b9426d, 0x8ee7d30c9b4bab84,
            0x1bcd1f840d3ed0f0, 0xcd14c6f310abdc3d, 0x69d558ed64d6a25f};
    b[0] = f12cn::zero_v;

    mul<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(ret[0] == b[0]);
  }

  SECTION("of pre-computed values returns expected value") {
    a[0] = {0x0397a38320170cd4, 0x734c1b2c9e761d30, 0x5ed255ad9a48beb5,
            0x095a3c6b22a7fcfc, 0x2294ce75d4e26a27, 0x13338bd870011ebb};
    b[0] = {0xb9c3c7c5b1196af7, 0x2580e2086ce335c1, 0xf49aed3d8a57ef42,
            0x41f281e49846e878, 0xe0762346c38452ce, 0x0652e89326e57dc0};
    constexpr f12t::element expected{0xf96ef3d711ab5355, 0xe8d459ea00f148dd, 0x53f7354a5f00fa78,
                                     0x9e34a4f3125c5f83, 0x3fbe0c47ca74c19e, 0x01b06a8bbd4adfe4};

    mul<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(expected == ret[0]);
  }

  SECTION("matches a random host output") {
    basn::fast_random_number_generator rng{1, 2};
    f12rn::generate_random_element(a[0], rng);
    f12rn::generate_random_element(b[0], rng);
    const f12t::element a_host = a[0];
    const f12t::element b_host = b[0];
    f12t::element ret_host;

    mul(ret_host, a_host, b_host);

    mul<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(ret_host == ret[0]);
  }
}
