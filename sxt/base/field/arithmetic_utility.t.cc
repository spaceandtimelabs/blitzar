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
#include "sxt/base/field/arithmetic_utility.h"

#include <random>

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::basfld;

__global__ void adc(uint64_t* __restrict__ ret, uint64_t* __restrict__ carry,
                    const uint64_t* __restrict__ a, const uint64_t* __restrict__ b,
                    const uint64_t* __restrict__ c) {
  adc(ret[0], carry[0], a[0], b[0], c[0]);
}

__global__ void mac(uint64_t* __restrict__ ret, uint64_t* __restrict__ carry,
                    const uint64_t* __restrict__ a, const uint64_t* __restrict__ b,
                    const uint64_t* __restrict__ c) {
  mac(ret[0], carry[0], a[0], b[0], c[0]);
}

TEST_CASE("mac (multiplication and carry) can handle computation") {
  SECTION("with minimum values") {
    constexpr uint64_t a{0x0};
    constexpr uint64_t b{0x0};
    constexpr uint64_t c{0x0};
    uint64_t carry{0x0};
    uint64_t ret{0x0};
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0x0);
    REQUIRE(carry == 0x0);
  }

  SECTION("without carryover on pre-comuputed values") {
    constexpr uint64_t a{0x1};
    constexpr uint64_t b{0x2};
    constexpr uint64_t c{0x3};
    uint64_t carry{0x4};
    uint64_t ret{0x0};
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0xb);
    REQUIRE(carry == 0x0);
  }

  SECTION("with carryover on pre-comuputed values") {
    constexpr uint64_t a{0xb9feffffffffaaab};
    constexpr uint64_t b{0x1eabfffeb153ffff};
    constexpr uint64_t c{0x6730d2a0f6b0f624};
    uint64_t carry{0x64774b84f38512bf};
    uint64_t ret{0x0};
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0xc974d8dba4a3c746);
    REQUIRE(carry == 0xc5d0d7bda277422);
  }

  SECTION("with maximum values") {
    constexpr uint64_t a{0xffffffffffffffff};
    constexpr uint64_t b{0xffffffffffffffff};
    constexpr uint64_t c{0xffffffffffffffff};
    uint64_t carry{0xffffffffffffffff};
    uint64_t ret{0x0};
    mac(ret, carry, a, b, c);
    REQUIRE(ret == 0xffffffffffffffff);
    REQUIRE(carry == 0xffffffffffffffff);
  }
}

TEST_CASE("mac (multiplication and carry) can handle computation on device") {
  memmg::managed_array<uint64_t> a(1, memr::get_managed_device_resource());
  memmg::managed_array<uint64_t> b(1, memr::get_managed_device_resource());
  memmg::managed_array<uint64_t> c(1, memr::get_managed_device_resource());
  memmg::managed_array<uint64_t> ret(1, memr::get_managed_device_resource());
  memmg::managed_array<uint64_t> carry(1, memr::get_managed_device_resource());

  SECTION("with minimum values") {
    a[0] = 0;
    b[0] = 0;
    c[0] = 0;
    carry[0] = 0;
    ret[0] = 0;

    mac<<<1, 1>>>(ret.data(), carry.data(), a.data(), b.data(), c.data());
    cudaDeviceSynchronize();

    REQUIRE(ret[0] == 0);
    REQUIRE(carry[0] == 0x0);
  }

  SECTION("without carryover on pre-comuputed values") {
    a[0] = 0x1;
    b[0] = 0x2;
    c[0] = 0x3;
    carry[0] = 0x4;
    ret[0] = 0x0;

    mac<<<1, 1>>>(ret.data(), carry.data(), a.data(), b.data(), c.data());
    cudaDeviceSynchronize();

    REQUIRE(ret[0] == 0xb);
    REQUIRE(carry[0] == 0x0);
  }

  SECTION("with carryover on pre-comuputed values") {
    a[0] = 0xb9feffffffffaaab;
    b[0] = 0x1eabfffeb153ffff;
    c[0] = 0x6730d2a0f6b0f624;
    carry[0] = 0x64774b84f38512bf;
    ret[0] = 0x0;

    mac<<<1, 1>>>(ret.data(), carry.data(), a.data(), b.data(), c.data());
    cudaDeviceSynchronize();

    REQUIRE(ret[0] == 0xc974d8dba4a3c746);
    REQUIRE(carry[0] == 0xc5d0d7bda277422);
  }

  SECTION("with maximum values") {
    a[0] = 0xffffffffffffffff;
    b[0] = 0xffffffffffffffff;
    c[0] = 0xffffffffffffffff;
    carry[0] = 0xffffffffffffffff;
    ret[0] = 0x0;

    mac<<<1, 1>>>(ret.data(), carry.data(), a.data(), b.data(), c.data());
    cudaDeviceSynchronize();

    REQUIRE(ret[0] == 0xffffffffffffffff);
    REQUIRE(carry[0] == 0xffffffffffffffff);
  }

  SECTION("by matching the non-GPU implementation on a random value") {
    basn::fast_random_number_generator rng{1, 2};

    a[0] = rng();
    b[0] = rng();
    c[0] = rng();
    carry[0] = rng();
    ret[0] = 0;

    uint64_t ret_expected{0};
    uint64_t carry_expected{carry[0]};
    mac(ret_expected, carry_expected, a[0], b[0], c[0]);

    mac<<<1, 1>>>(ret.data(), carry.data(), a.data(), b.data(), c.data());
    cudaDeviceSynchronize();

    REQUIRE(ret[0] == ret_expected);
    REQUIRE(carry[0] == carry_expected);
  }
}

TEST_CASE("adc (addition and carry) can handle computation") {
  SECTION("with minimum values") {
    constexpr uint64_t a{0x0};
    constexpr uint64_t b{0x0};
    uint64_t carry{0x0};
    uint64_t ret{0x0};
    adc(ret, carry, a, b, carry);
    REQUIRE(ret == 0x0);
    REQUIRE(carry == 0x0);
  }

  SECTION("without carryover on pre-comuputed values") {
    constexpr uint64_t a{0x3};
    constexpr uint64_t b{0x4};
    uint64_t carry{0xa};
    uint64_t ret{0x0};
    adc(ret, carry, a, b, carry);
    REQUIRE(ret == 0x11);
    REQUIRE(carry == 0x0);
  }

  SECTION("with carryover on pre-comuputed values") {
    constexpr uint64_t a{0x1};
    constexpr uint64_t b{0xffffffffffffffff};
    uint64_t carry{0x0};
    uint64_t ret{0x0};
    adc(ret, carry, a, b, carry);
    REQUIRE(ret == 0x0);
    REQUIRE(carry == 0x1);
  }

  SECTION("with maximum values") {
    constexpr uint64_t a{0xffffffffffffffff};
    constexpr uint64_t b{0xffffffffffffffff};
    uint64_t carry{0xffffffffffffffff};
    uint64_t ret{0x0};
    adc(ret, carry, a, b, carry);
    REQUIRE(ret == 0xfffffffffffffffd);
    REQUIRE(carry == 0x2);
  }
}

TEST_CASE("adc (addition and carry) can handle computation on device") {
  memmg::managed_array<uint64_t> a(1, memr::get_managed_device_resource());
  memmg::managed_array<uint64_t> b(1, memr::get_managed_device_resource());
  memmg::managed_array<uint64_t> ret(1, memr::get_managed_device_resource());
  memmg::managed_array<uint64_t> carry(1, memr::get_managed_device_resource());

  SECTION("with minimum values on the GPU") {
    a[0] = 0x0;
    b[0] = 0x0;
    carry[0] = 0x0;
    ret[0] = 0x0;

    adc<<<1, 1>>>(ret.data(), carry.data(), a.data(), b.data(), carry.data());
    cudaDeviceSynchronize();

    REQUIRE(ret[0] == 0x0);
    REQUIRE(carry[0] == 0x0);
  }

  SECTION("without carryover on pre-comuputed values") {
    a[0] = 0x3;
    b[0] = 0x4;
    carry[0] = 0xa;
    ret[0] = 0x0;

    adc<<<1, 1>>>(ret.data(), carry.data(), a.data(), b.data(), carry.data());
    cudaDeviceSynchronize();

    REQUIRE(ret[0] == 0x11);
    REQUIRE(carry[0] == 0x0);
  }

  SECTION("with carryover on pre-comuputed values") {
    a[0] = 0x1;
    b[0] = 0xffffffffffffffff;
    carry[0] = 0x0;
    ret[0] = 0x0;

    adc<<<1, 1>>>(ret.data(), carry.data(), a.data(), b.data(), carry.data());
    cudaDeviceSynchronize();

    REQUIRE(ret[0] == 0x0);
    REQUIRE(carry[0] == 0x1);
  }

  SECTION("by matching the non-GPU implementation on random values") {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> uni_dis;
    std::bernoulli_distribution ber_d(0.5);

    for (unsigned i = 0; i < 100; ++i) {
      a[0] = uni_dis(gen);
      b[0] = uni_dis(gen);
      carry[0] = ber_d(gen) ? 0x0 : 0x1;
      ret[0] = 0x0;

      uint64_t ret_expected{0};
      uint64_t carry_expected{0};
      adc(ret_expected, carry_expected, a[0], b[0], carry[0]);

      adc<<<1, 1>>>(ret.data(), carry.data(), a.data(), b.data(), carry.data());
      cudaDeviceSynchronize();

      REQUIRE(ret[0] == ret_expected);
      REQUIRE(carry[0] == carry_expected);
    }
  }

  SECTION("with maximum values") {
    a[0] = 0xffffffffffffffff;
    b[0] = 0xfffffffffffffffe;
    carry[0] = 0x1;
    ret[0] = 0x0;

    adc<<<1, 1>>>(ret.data(), carry.data(), a.data(), b.data(), carry.data());
    cudaDeviceSynchronize();

    REQUIRE(ret[0] == 0xfffffffffffffffe);
    REQUIRE(carry[0] == 0x1);
  }
}

TEST_CASE("sbb (subtraction and borrow) can handle computation") {
  SECTION("with left summand less than than right summand and no borrow") {
    constexpr uint64_t a{0x031ff1ffffffffff};
    constexpr uint64_t b{0xeffff3ff5ff2feff};
    uint64_t borrow{0};
    uint64_t ret{0};
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 0x131ffe00a00d0100);
    REQUIRE(borrow == 0xffffffffffffffff);
  }

  SECTION("with left summand less than than right summand and borrow") {
    constexpr uint64_t a{0x031ff1f7f13ff4f2};
    constexpr uint64_t b{0xe078f3ff5ff2fef9};
    uint64_t borrow{0xdd5902076eb30a06};
    uint64_t ret{0};
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 0x22a6fdf8914cf5f8);
    REQUIRE(borrow == 0xffffffffffffffff);
  }

  SECTION("with left summand greater than right summand and no borrow") {
    constexpr uint64_t a{0xeffff3ff5ff2feff};
    constexpr uint64_t b{0x031ff1ffffffffff};
    uint64_t borrow{0};
    uint64_t ret{0};
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 0xece001ff5ff2ff00);
    REQUIRE(borrow == 0);
  }

  SECTION("with left summand greater than right summand and borrow") {
    constexpr uint64_t a{0xe078f3ff5ff2fef9};
    constexpr uint64_t b{0x031ff1f7f13ff4f2};
    uint64_t borrow{0xdd5902076eb30a06};
    uint64_t ret{0};
    sbb(ret, borrow, a, b);
    REQUIRE(ret == 0xdd5902076eb30a06);
    REQUIRE(borrow == 0x0);
  }
}
