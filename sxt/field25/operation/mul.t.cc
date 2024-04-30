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
#include "sxt/field25/operation/mul.h"

#include "sxt/base/device/synchronization.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field25/base/constants.h"
#include "sxt/field25/base/montgomery.h"
#include "sxt/field25/constant/zero.h"
#include "sxt/field25/random/element.h"
#include "sxt/field25/type/element.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::f25o;

__global__ void mul(f25t::element* __restrict__ ret, const f25t::element* __restrict__ a,
                    const f25t::element* __restrict__ b) {
  mul(ret[0], a[0], b[0]);
}

TEST_CASE("multiplication") {
  SECTION("of a random field element and zero returns zero") {
    f25t::element a;
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(a, rng);
    f25t::element ret;

    mul(ret, a, f25cn::zero_v);

    REQUIRE(ret == f25cn::zero_v);
  }

  SECTION("of one with itself returns one") {
    constexpr f25t::element one{f25b::r_v.data()};
    f25t::element ret;

    mul(ret, one, one);

    REQUIRE(one == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random bn254 base field element generated using the SAGE library.
    constexpr f25t::element a{0x5d19786fd5f59520, 0xa80980aec93386cf, 0x6f49109c5fb69712,
                              0x19803c04269ae364};
    constexpr f25t::element b{0x0746f2e0932048bf, 0xc05bea62ab71831e, 0x1342f6ebbc9497e9,
                              0x12b50c2429b1a851};
    constexpr f25t::element expected{0x241cac338ef8e513, 0x98220ca91953d8c1, 0x0bf0a5fd342762a0,
                                     0x0e616e612ed86a67};
    f25t::element ret;

    mul(ret, a, b);

    REQUIRE(expected == ret);
  }
}

TEST_CASE("multiplication on device") {
  memmg::managed_array<f25t::element> a(1, memr::get_managed_device_resource());
  memmg::managed_array<f25t::element> b(1, memr::get_managed_device_resource());
  memmg::managed_array<f25t::element> ret(1, memr::get_managed_device_resource());

  SECTION("of a random value and zero returns zero") {
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(a[0], rng);
    b[0] = f25cn::zero_v;

    mul<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(ret[0] == b[0]);
  }

  SECTION("of one with itself returns one") {
    a[0] = f25b::r_v.data();

    mul<<<1, 1>>>(ret.data(), a.data(), a.data());
    basdv::synchronize_device();

    REQUIRE(a[0] == ret[0]);
  }  

  SECTION("of pre-computed values returns expected value") {
    // Random bn254 base field element generated using the SAGE library.
    a[0] = {0x5d19786fd5f59520, 0xa80980aec93386cf, 0x6f49109c5fb69712,
                              0x19803c04269ae364};
    b[0] = {0x0746f2e0932048bf, 0xc05bea62ab71831e, 0x1342f6ebbc9497e9,
                              0x12b50c2429b1a851};
    constexpr f25t::element expected{0x241cac338ef8e513, 0x98220ca91953d8c1, 0x0bf0a5fd342762a0,
                                     0x0e616e612ed86a67};                                     

    mul<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(expected == ret[0]);
  }

  SECTION("matches a random host output") {
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(a[0], rng);
    f25rn::generate_random_element(b[0], rng);
    const f25t::element a_host = a[0];
    const f25t::element b_host = b[0];
    f25t::element ret_host;

    mul(ret_host, a_host, b_host);

    mul<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(ret_host == ret[0]);
  }
}
