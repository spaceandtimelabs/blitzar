/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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
#include "sxt/multiexp/bucket_method/reduce.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we reduce bucket sums into a multiexponentiation result") {
  using E = bascrv::element97;

  std::vector<E> res(1);
  std::vector<E> expected(1);
  std::pmr::vector<E> sums{8, memr::get_managed_device_resource()};

  SECTION("we can reduce a non-zero bucket in the identity position") {
    sums[0] = 33u;
    auto fut = reduce_buckets<E>(res, sums, 1, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {33u};
    REQUIRE(res == expected);
  }

  SECTION("we handle multiple outputs") {
    res.resize(2);
    sums.resize(16);
    sums[0] = 33u;
    sums[8] = 77u;
    auto fut = reduce_buckets<E>(res, sums, 1, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {33u, 77u};
    REQUIRE(res == expected);
  }

  SECTION("we can reduce a non-zero bucket in second position") {
    sums[1] = 33u;
    auto fut = reduce_buckets<E>(res, sums, 1, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {66u};
    REQUIRE(res == expected);
  }

  SECTION("we handle bit widths greater than 1") {
    sums.resize(3 * 4);
    sums[0] = 33u;
    auto fut = reduce_buckets<E>(res, sums, 1, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {33u};
    REQUIRE(res == expected);
  }

  SECTION("we can reduce a non-zero bucket in the second position with bit_width = 2") {
    sums.resize(3 * 4);
    sums[1] = 33u;
    auto fut = reduce_buckets<E>(res, sums, 1, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {66u};
    REQUIRE(res == expected);
  }

  SECTION("we can reduce a non-zero bucket in the third position with bit_width = 2") {
    sums.resize(3 * 4);
    sums[2] = 33u;
    auto fut = reduce_buckets<E>(res, sums, 1, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {99u};
    REQUIRE(res == expected);
  }
}
