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
#include "sxt/field25/base/subtract_p.h"

#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field25/base/constants.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::f25b;

__global__ void subtract_p_device(uint64_t* __restrict__ ret, const uint64_t* __restrict__ a) {
  subtract_p(&ret[0], &a[0]);
}

TEST_CASE("subtract_p (subtraction with the modulus) can handle computation") {
  SECTION("of zero") {
    constexpr std::array<uint64_t, 4> a = {0, 0, 0, 0};
    std::array<uint64_t, 4> ret;

    subtract_p(ret.data(), a.data());

    REQUIRE(ret == a);
  }

  SECTION("of one below the modulus p_v") {
    constexpr std::array<uint64_t, 4> a = {0x3c208c16d87cfd46, 0x97816a916871ca8d,
                                           0xb85045b68181585d, 0x30644e72e131a029};
    std::array<uint64_t, 4> ret;

    subtract_p(ret.data(), a.data());

    REQUIRE(ret == a);
  }

  SECTION("of the modulus p_v") {
    constexpr std::array<uint64_t, 4> expect = {0, 0, 0, 0};
    std::array<uint64_t, 4> ret;

    subtract_p(ret.data(), p_v.data());

    REQUIRE(expect == ret);
  }

  SECTION("of one above the modulus p_v") {
    constexpr std::array<uint64_t, 4> a = {0x3c208c16d87cfd48, 0x97816a916871ca8d,
                                           0xb85045b68181585d, 0x30644e72e131a029};
    constexpr std::array<uint64_t, 4> expect = {1, 0, 0, 0};
    std::array<uint64_t, 4> ret;

    subtract_p(ret.data(), a.data());

    REQUIRE(expect == ret);
  }
}

TEST_CASE("subtract_p (subtraction with the modulus) can handle on device computation") {
  memmg::managed_array<uint64_t> a(4, memr::get_managed_device_resource());
  memmg::managed_array<uint64_t> ret(4, memr::get_managed_device_resource());
  
  SECTION("of zero") {
    std::fill(a.begin(), a.end(), 0);

    subtract_p_device<<<1,1>>>(ret.data(), a.data());
    basdv::synchronize_device();

    for (unsigned i = 0; i < 4; ++i) {
      REQUIRE(ret[i] == a[i]);
    }
  }

  SECTION("of one below the modulus p_v") {
    a[0] = 0x3c208c16d87cfd46;
    a[1] = 0x97816a916871ca8d;
    a[2] = 0xb85045b68181585d;
    a[3] = 0x30644e72e131a029;
    
    subtract_p_device<<<1,1>>>(ret.data(), a.data());
    basdv::synchronize_device();

    for (unsigned i = 0; i < 4; ++i) {
      REQUIRE(ret[i] == a[i]);
    }
  }

  SECTION("of the modulus p_v") {
    constexpr std::array<uint64_t, 4> expect = {0, 0, 0, 0};
    for (unsigned i = 0; i < 4; ++i) {
      a[i] = p_v[i];
    }

    subtract_p_device<<<1,1>>>(ret.data(), a.data());
    basdv::synchronize_device();

    for (unsigned i = 0; i < 4; ++i) {
      REQUIRE(expect[i] == ret[i]);
    }
  }

  SECTION("of one above the modulus p_v") {
    a[0] = 0x3c208c16d87cfd48;
    a[1] = 0x97816a916871ca8d;
    a[2] = 0xb85045b68181585d;
    a[3] = 0x30644e72e131a029;
    constexpr std::array<uint64_t, 4> expect = {1, 0, 0, 0};

    subtract_p_device<<<1,1>>>(ret.data(), a.data());
    basdv::synchronize_device();

    for (unsigned i = 0; i < 4; ++i) {
      REQUIRE(expect[i] == ret[i]);
    }
  }

  SECTION("matches a random host output") {
    a[0] = 0x3c208c16d87cfd48;
    a[1] = 0x97816a916871ca8d;
    a[2] = 0xb85045b68181585d;
    a[3] = 0x30644e72e131a029;
    constexpr std::array<uint64_t, 4> expect = {1, 0, 0, 0};

    subtract_p_device<<<1,1>>>(ret.data(), a.data());
    basdv::synchronize_device();

    for (unsigned i = 0; i < 4; ++i) {
      REQUIRE(expect[i] == ret[i]);
    }
  } 
}
