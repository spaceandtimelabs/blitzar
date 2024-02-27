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
#include "sxt/multiexp/bucket_method2/sum.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxbk2;

TEST_CASE("we can compute the bucket sums for a chunk") {
  using E = bascrv::element97;
  const unsigned element_num_bytes = 32;
  const unsigned bit_width = 8;

  std::pmr::vector<E> sums(255 * 32, memr::get_managed_device_resource());
  std::vector<E> generators = {3u};
  std::vector<const uint8_t*> scalars;

  std::pmr::vector<E> expected(sums.size());

  SECTION("we can compute bucket sums for a single exponent of zero") {
    std::vector<uint8_t> scalars1(32);
    scalars = {scalars1.data()};
    auto fut = sum_buckets<E>(sums, generators, scalars, element_num_bytes, bit_width);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(sums == expected);
  }

  SECTION("we can compute the bucket sums for a single exponent of one") {
    std::vector<uint8_t> scalars1(32);
    scalars1[0] = 1;
    scalars = {scalars1.data()};
    auto fut = sum_buckets<E>(sums, generators, scalars, element_num_bytes, bit_width);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = generators[0];
    REQUIRE(sums == expected);
  }

  SECTION("we can compute the bucket sums for two exponents of one") {
    generators = {3u, 4u};
    std::vector<uint8_t> scalars1(64);
    scalars1[0] = 1;
    scalars1[32] = 1;
    scalars = {scalars1.data()};
    auto fut = sum_buckets<E>(sums, generators, scalars, element_num_bytes, bit_width);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = generators[0].value + generators[1].value;
    REQUIRE(sums == expected);
  }
}
