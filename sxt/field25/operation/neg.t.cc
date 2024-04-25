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
#include "sxt/field25/operation/neg.h"

#include "sxt/base/device/synchronization.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field25/base/constants.h"
#include "sxt/field25/constant/zero.h"
#include "sxt/field25/random/element.h"
#include "sxt/field25/type/element.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::f25o;

__global__ void neg(f25t::element* __restrict__ ret, const f25t::element* __restrict__ a) {
  neg(ret[0], a[0]);
}

TEST_CASE("negation") {
  SECTION("of zero and the modulus are equal") {
    f25t::element ret_zero;
    f25t::element ret_modulus;

    neg(ret_zero, f25cn::zero_v);
    neg(ret_modulus, f25b::p_v.data());

    REQUIRE(ret_zero == ret_modulus);
  }

  SECTION("of the modulus minus one is one") {
    constexpr f25t::element modulus_minus_one{0x3c208c16d87cfd46, 0x97816a916871ca8d,
                                              0xb85045b68181585d, 0x30644e72e131a029};
    constexpr f25t::element one{1, 0, 0, 0};
    f25t::element ret;

    neg(ret, modulus_minus_one);

    REQUIRE(ret == one);
  }

  SECTION("of a pre-computed value is expected") {
    // Random bn254 base field element generated using the SAGE library.
    constexpr f25t::element a{0x1149c21473b043fd, 0x5610b2a5c08c7ecf, 0xc9e31f2d914c45b5,
                              0x066031eb7a3ca7fd};
    constexpr f25t::element expected{0x2ad6ca0264ccb94a, 0x4170b7eba7e54bbe, 0xee6d2688f03512a8,
                                     0x2a041c8766f4f82b};
    f25t::element ret;

    neg(ret, a);

    REQUIRE(expected == ret);
  }
}

TEST_CASE("negation on the device") {
  memmg::managed_array<f25t::element> a(1, memr::get_managed_device_resource());
  memmg::managed_array<f25t::element> ret(1, memr::get_managed_device_resource());

  SECTION("of zero and the modulus are equal") {
    memmg::managed_array<f25t::element> p(1, memr::get_managed_device_resource());
    memmg::managed_array<f25t::element> ret_zero(1, memr::get_managed_device_resource());

    a[0] = f25cn::zero_v.data();
    p[0] = f25b::p_v.data();

    neg<<<1, 1>>>(ret_zero.data(), a.data());
    neg<<<1, 1>>>(ret.data(), p.data());
    basdv::synchronize_device();

    REQUIRE(ret_zero[0] == ret[0]);
  }

  SECTION("of the modulus minus one is one") {
    a[0] = {0x3c208c16d87cfd46, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029};
    constexpr f25t::element one{1, 0, 0, 0};

    neg<<<1, 1>>>(ret.data(), a.data());
    basdv::synchronize_device();

    REQUIRE(one == ret[0]);
  }

  SECTION("of a pre-computed value is expected") {
    // Random bn254 base field element generated using the SAGE library.
    a[0] = {0x1149c21473b043fd, 0x5610b2a5c08c7ecf, 0xc9e31f2d914c45b5, 0x066031eb7a3ca7fd};
    constexpr f25t::element expected{0x2ad6ca0264ccb94a, 0x4170b7eba7e54bbe, 0xee6d2688f03512a8,
                                     0x2a041c8766f4f82b};

    neg<<<1, 1>>>(ret.data(), a.data());
    basdv::synchronize_device();

    REQUIRE(expected == ret[0]);
  }

  SECTION("matches a random host output") {
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(a[0], rng);
    const f25t::element a_host = a[0];
    f25t::element ret_host;

    neg(ret_host, a_host);

    neg<<<1, 1>>>(ret.data(), a.data());
    basdv::synchronize_device();

    REQUIRE(ret_host == ret[0]);
  }
}
