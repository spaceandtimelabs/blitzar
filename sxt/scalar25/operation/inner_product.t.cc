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
#include "sxt/scalar25/operation/inner_product.h"

#include <random>
#include <vector>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/random/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::s25o;
using sxt::s25t::operator""_s25;

static void make_dataset(memmg::managed_array<s25t::element>& a_host,
                         memmg::managed_array<s25t::element>& b_host,
                         memmg::managed_array<s25t::element>& a_dev,
                         memmg::managed_array<s25t::element>& b_dev,
                         basn::fast_random_number_generator& rng, size_t n) noexcept {
  a_host = memmg::managed_array<s25t::element>(n);
  b_host = memmg::managed_array<s25t::element>(n);
  a_dev = memmg::managed_array<s25t::element>(n, memr::get_device_resource());
  b_dev = memmg::managed_array<s25t::element>(n, memr::get_device_resource());
  s25rn::generate_random_elements(a_host, rng);
  s25rn::generate_random_elements(b_host, rng);
  basdv::memcpy_host_to_device(a_dev.data(), a_host.data(), n * sizeof(s25t::element));
  basdv::memcpy_host_to_device(b_dev.data(), b_host.data(), n * sizeof(s25t::element));
}

TEST_CASE("we can compute the inner product of two scalar vectors") {
  s25t::element res;

  SECTION("we properly handle an inner product of vectors consisting of only a single element") {
    std::vector<s25t::element> lhs = {0x3_s25};
    std::vector<s25t::element> rhs = {0x2_s25};
    inner_product(res, lhs, rhs);
    REQUIRE(res == 0x6_s25);
  }

  SECTION("we handle vectors of more than a single element") {
    std::vector<s25t::element> lhs = {0x3_s25, 0x123_s25, 0x456_s25};
    std::vector<s25t::element> rhs = {0x2_s25, 0x9234_s25, 0x6435_s25};
    inner_product(res, lhs, rhs);
    auto expected = lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
    REQUIRE(res == expected);
  }

  SECTION("if vectors are of unequal length, we compute the product as if the smaller vector was "
          "padded with zeros") {
    std::vector<s25t::element> lhs = {0x3_s25, 0x123_s25, 0x456_s25};
    std::vector<s25t::element> rhs = {0x2_s25};
    inner_product(res, lhs, rhs);
    auto expected = lhs[0] * rhs[0];
    REQUIRE(res == expected);
    inner_product(res, rhs, lhs);
    REQUIRE(res == expected);
  }
}

TEST_CASE("we can compute inner products asynchronously on the GPU") {
  memmg::managed_array<s25t::element> a_host, b_host;
  memmg::managed_array<s25t::element> a_dev{memr::get_device_resource()},
      b_dev{memr::get_device_resource()};
  basn::fast_random_number_generator rng{1, 2};

  SECTION("async inner product computes the same result as the host version for select "
          "lengths") {
    for (size_t n : {1, 63, 64, 65, 127, 128, 129, 255, 256, 257}) {
      make_dataset(a_host, b_host, a_dev, b_dev, rng, n);
      auto res = async_inner_product(a_dev, b_dev);
      s25t::element expected_res;
      inner_product(expected_res, a_host, b_host);
      xens::get_scheduler().run();
      REQUIRE(res.value() == expected_res);
    }
  }

  SECTION("async inner product computes the same result as the host version for data of random "
          "length") {
    for (int i = 0; i < 10; ++i) {
      auto n = static_cast<size_t>(rng()) % 10'000u;
      make_dataset(a_host, b_host, a_dev, b_dev, rng, n);
      auto res = async_inner_product(a_dev, b_dev);
      s25t::element expected_res;
      inner_product(expected_res, a_host, b_host);
      xens::get_scheduler().run();
      REQUIRE(res.value() == expected_res);
    }
  }

  SECTION("async inner product handles spans of different lengths") {
    size_t n = 100;
    size_t m = 91;
    make_dataset(a_host, b_host, a_dev, b_dev, rng, n);
    auto res = async_inner_product(a_dev, {b_dev.data(), m});
    s25t::element expected_res;
    inner_product(expected_res, a_host, {b_host.data(), m});
    xens::get_scheduler().run();
    REQUIRE(res.value() == expected_res);
  }

  SECTION("async inner product works with both device and host points") {
    size_t n = 100;
    make_dataset(a_host, b_host, a_dev, b_dev, rng, n);
    auto res1 = async_inner_product(a_dev, b_host);
    auto res2 = async_inner_product(a_host, b_dev);
    auto res3 = async_inner_product(a_host, b_host);
    s25t::element expected_res;
    inner_product(expected_res, a_host, b_host);
    xens::get_scheduler().run();
    REQUIRE(res1.value() == expected_res);
    REQUIRE(res2.value() == expected_res);
    REQUIRE(res3.value() == expected_res);
  }
}
