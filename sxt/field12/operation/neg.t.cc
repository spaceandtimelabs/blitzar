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
#include "sxt/field12/operation/neg.h"

#include "sxt/base/device/synchronization.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field12/base/constants.h"
#include "sxt/field12/constant/zero.h"
#include "sxt/field12/random/element.h"
#include "sxt/field12/type/element.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::f12o;

__global__ void neg(f12t::element* __restrict__ ret, const f12t::element* __restrict__ a) {
  neg(ret[0], a[0]);
}

TEST_CASE("negation") {
  SECTION("of zero and the modulus are equal") {
    f12t::element ret_zero;
    f12t::element ret_modulus;

    neg(ret_zero, f12cn::zero_v);
    neg(ret_modulus, f12b::p_v.data());

    REQUIRE(ret_zero == ret_modulus);
  }

  SECTION("of the modulus minus one is one") {
    constexpr f12t::element modulus_minus_one{0xb9feffffffffaaaa, 0x1eabfffeb153ffff,
                                              0x6730d2a0f6b0f624, 0x64774b84f38512bf,
                                              0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
    constexpr f12t::element one{0x1, 0x0, 0x0, 0x0, 0x0, 0x0};
    f12t::element ret;

    neg(ret, modulus_minus_one);

    REQUIRE(ret == one);
  }

  SECTION("of a pre-computed value is expected") {
    constexpr f12t::element a{0x5360bb5978678032, 0x7dd275ae799e128e, 0x5c5b5071ce4f4dcf,
                              0xcdb21f93078dbb3e, 0xc32365c5e73f474a, 0x115a2a5489babe5b};
    constexpr f12t::element expected{0x669e44a687982a79, 0xa0d98a5037b5ed71, 0x0ad5822f2861a854,
                                     0x96c52bf1ebf75781, 0x87f841f05c0c658c, 0x08a6e795afc5283e};
    f12t::element ret;

    neg(ret, a);

    REQUIRE(expected == ret);
  }
}

TEST_CASE("negation on the device") {
  memmg::managed_array<f12t::element> a(1, memr::get_managed_device_resource());
  memmg::managed_array<f12t::element> ret(1, memr::get_managed_device_resource());

  SECTION("of zero and the modulus are equal") {
    memmg::managed_array<f12t::element> p(1, memr::get_managed_device_resource());
    memmg::managed_array<f12t::element> ret_zero(1, memr::get_managed_device_resource());

    a[0] = f12cn::zero_v.data();
    p[0] = f12b::p_v.data();

    neg<<<1, 1>>>(ret_zero.data(), a.data());
    neg<<<1, 1>>>(ret.data(), p.data());
    basdv::synchronize_device();

    REQUIRE(ret_zero[0] == ret[0]);
  }

  SECTION("of the modulus minus one is one") {
    a[0] = {0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624,
            0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
    constexpr f12t::element one{0x1, 0x0, 0x0, 0x0, 0x0, 0x0};

    neg<<<1, 1>>>(ret.data(), a.data());
    basdv::synchronize_device();

    REQUIRE(one == ret[0]);
  }

  SECTION("of a pre-computed value is expected") {
    a[0] = {0x5360bb5978678032, 0x7dd275ae799e128e, 0x5c5b5071ce4f4dcf,
            0xcdb21f93078dbb3e, 0xc32365c5e73f474a, 0x115a2a5489babe5b};
    constexpr f12t::element expected{0x669e44a687982a79, 0xa0d98a5037b5ed71, 0x0ad5822f2861a854,
                                     0x96c52bf1ebf75781, 0x87f841f05c0c658c, 0x08a6e795afc5283e};

    neg<<<1, 1>>>(ret.data(), a.data());
    basdv::synchronize_device();

    REQUIRE(expected == ret[0]);
  }

  SECTION("matches a random host output") {
    basn::fast_random_number_generator rng{1, 2};
    f12rn::generate_random_element(a[0], rng);
    const f12t::element a_host = a[0];
    f12t::element ret_host;

    neg(ret_host, a_host);

    neg<<<1, 1>>>(ret.data(), a.data());
    basdv::synchronize_device();

    REQUIRE(ret_host == ret[0]);
  }
}
