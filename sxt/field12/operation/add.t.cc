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
#include "sxt/field12/operation/add.h"

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

__global__ void add(f12t::element* __restrict__ ret, const f12t::element* __restrict__ a,
                    const f12t::element* __restrict__ b) {
  add(ret[0], a[0], b[0]);
}

TEST_CASE("addition") {
  SECTION("of pre-computed value and zero returns pre-computed value") {
    f12t::element a;
    basn::fast_random_number_generator rng{1, 2};
    f12rn::generate_random_element(a, rng);

    f12t::element ret;
    add(ret, a, f12cn::zero_v);

    REQUIRE(a == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random values between 1 and p_v generated using the SAGE library.
    constexpr f12t::element a{0xd13cd59bb4634e05, 0x0f2509b85592e56f, 0x651937e06008a619,
                              0x64d5bd872b39c8a7, 0x841f0f8892fa4f10, 0x78048ec7ecc6399};
    constexpr f12t::element b{0x16c9e69b4dea3f04, 0xf0a6ec99d0b1f9be, 0x22bb437b9b63365f,
                              0x1c46c6dd44489804, 0x10f8e4ed03d5659a, 0x14fe816ddb9e6192};
    constexpr f12t::element expected{0x2e07bc37024de25e, 0xe11ff65374f0df2e, 0x20a3a8bb04bae654,
                                     0x1ca538df7bfd4dec, 0x49fc4cbf538407d3, 0x27db87020eade91};

    f12t::element ret;
    add(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("of the modulus minus one and one returns zero") {
    constexpr f12t::element a{0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624,
                              0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
    constexpr f12t::element b{0x1, 0x0, 0x0, 0x0, 0x0, 0x0};

    f12t::element ret;
    add(ret, a, b);

    REQUIRE(f12cn::zero_v == ret);
  }
}

TEST_CASE("addition on device") {
  memmg::managed_array<f12t::element> a(1, memr::get_managed_device_resource());
  memmg::managed_array<f12t::element> b(1, memr::get_managed_device_resource());
  memmg::managed_array<f12t::element> ret(1, memr::get_managed_device_resource());

  SECTION("of pre-computed value and zero returns pre-computed value") {
    basn::fast_random_number_generator rng{1, 2};
    f12rn::generate_random_element(a[0], rng);
    b[0] = f12cn::zero_v;

    add<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(a[0] == ret[0]);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random values between 1 and p_v generated using the SAGE library.
    a[0] = {0xd13cd59bb4634e05, 0x0f2509b85592e56f, 0x651937e06008a619,
            0x64d5bd872b39c8a7, 0x841f0f8892fa4f10, 0x78048ec7ecc6399};
    b[0] = {0x16c9e69b4dea3f04, 0xf0a6ec99d0b1f9be, 0x22bb437b9b63365f,
            0x1c46c6dd44489804, 0x10f8e4ed03d5659a, 0x14fe816ddb9e6192};
    constexpr f12t::element expected{0x2e07bc37024de25e, 0xe11ff65374f0df2e, 0x20a3a8bb04bae654,
                                     0x1ca538df7bfd4dec, 0x49fc4cbf538407d3, 0x27db87020eade91};

    add<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(expected == ret[0]);
  }

  SECTION("of the modulus minus one and one returns zero") {
    a[0] = {0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624,
            0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
    b[0] = {0x1, 0x0, 0x0, 0x0, 0x0, 0x0};

    add<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(f12cn::zero_v == ret[0]);
  }

  SECTION("matches a random host output") {
    basn::fast_random_number_generator rng{1, 2};
    f12rn::generate_random_element(a[0], rng);
    f12rn::generate_random_element(b[0], rng);
    const f12t::element a_host = a[0];
    const f12t::element b_host = b[0];
    f12t::element ret_host;

    add(ret_host, a_host, b_host);

    add<<<1, 1>>>(ret.data(), a.data(), b.data());
    basdv::synchronize_device();

    REQUIRE(ret_host == ret[0]);
  }
}
