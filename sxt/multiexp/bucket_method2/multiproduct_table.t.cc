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
#include "sxt/multiexp/bucket_method2/multiproduct_table.h"

#include <algorithm>
#include <vector>

#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxbk2;

TEST_CASE("we can construct the multi-product table for the bucket method") {
  std::pmr::vector<uint16_t> bucket_prefix_counts{memr::get_managed_device_resource()};
  std::pmr::vector<uint16_t> indexes{memr::get_managed_device_resource()};
  std::vector<const uint8_t*> scalars;
  const unsigned element_num_bytes = 32;
  const unsigned bit_width = 8;
  const unsigned num_buckets_per_digit = 255;
  const unsigned num_digits = 32;
  bucket_prefix_counts.resize(num_buckets_per_digit * num_digits);
  unsigned n = 0;

  std::pmr::vector<uint16_t> expected_counts(bucket_prefix_counts.size());
  std::pmr::vector<uint16_t> expected_indexes;

  SECTION("we handle the case of a single scalar of zeros") {
    n = 1;
    std::vector<uint8_t> scalars1(32 * n);
    scalars = {scalars1.data()};
    indexes.resize(num_digits * n);
    auto fut = make_multiproduct_table(bucket_prefix_counts, indexes, scalars, element_num_bytes,
                                       bit_width, n);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected_indexes.resize(indexes.size());
    basdv::synchronize_device();
    REQUIRE(bucket_prefix_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }

  SECTION("we handle the case of a single scalar of one") {
    n = 1;
    std::vector<uint8_t> scalars1(32 * n);
    scalars1[0] = 1;
    scalars = {scalars1.data()};
    indexes.resize(num_digits * n);
    auto fut = make_multiproduct_table(bucket_prefix_counts, indexes, scalars, element_num_bytes,
                                       bit_width, n);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    std::fill_n(expected_counts.begin(), num_buckets_per_digit, 1u);
    expected_indexes.resize(indexes.size());
    basdv::synchronize_device();
    REQUIRE(bucket_prefix_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }

  SECTION("we handle two scalars of zero and one") {
    n = 2;
    std::vector<uint8_t> scalars1(32 * n);
    scalars1[0] = 0;
    scalars1[element_num_bytes] = 1;
    scalars = {scalars1.data()};
    indexes.resize(num_digits * n);
    auto fut = make_multiproduct_table(bucket_prefix_counts, indexes, scalars, element_num_bytes,
                                       bit_width, n);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    std::fill_n(expected_counts.begin(), num_buckets_per_digit, 1u);
    expected_indexes.resize(indexes.size());
    expected_indexes[0] = 1;
    basdv::synchronize_device();
    REQUIRE(bucket_prefix_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }

  SECTION("we handle two scalars of one") {
    n = 2;
    std::vector<uint8_t> scalars1(32 * n);
    scalars1[0] = 1;
    scalars1[element_num_bytes] = 1;
    scalars = {scalars1.data()};
    indexes.resize(num_digits * n);
    auto fut = make_multiproduct_table(bucket_prefix_counts, indexes, scalars, element_num_bytes,
                                       bit_width, n);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    std::fill_n(expected_counts.begin(), num_buckets_per_digit, 2u);
    expected_indexes.resize(indexes.size());
    expected_indexes[0] = 0;
    expected_indexes[1] = 1;
    basdv::synchronize_device();
    REQUIRE(bucket_prefix_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }

  SECTION("we handle two scalars of one and two") {
    n = 2;
    std::vector<uint8_t> scalars1(32 * n);
    scalars1[0] = 1;
    scalars1[element_num_bytes] = 2;
    scalars = {scalars1.data()};
    indexes.resize(num_digits * n);
    auto fut = make_multiproduct_table(bucket_prefix_counts, indexes, scalars, element_num_bytes,
                                       bit_width, n);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected_counts[0] = 1;
    std::fill_n(expected_counts.begin() + 1, num_buckets_per_digit - 1, 2u);
    expected_indexes.resize(indexes.size());
    expected_indexes[0] = 0;
    expected_indexes[1] = 1;
    basdv::synchronize_device();
    REQUIRE(bucket_prefix_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }
}
