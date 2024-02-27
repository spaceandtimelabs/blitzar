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
#include "sxt/multiexp/bucket_method2/multiproduct_table_kernel.h"

#include <vector>

#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxbk2;

TEST_CASE("we can compute a bucket decomposition") {
  std::pmr::vector<uint16_t> bucket_counts{memr::get_managed_device_resource()};
  std::pmr::vector<uint16_t> indexes{memr::get_managed_device_resource()};
  std::pmr::vector<uint8_t> bytes{memr::get_managed_device_resource()};

  std::pmr::vector<uint16_t> expected_counts;
  std::pmr::vector<uint16_t> expected_indexes;

  SECTION("we handle the case of a single element of 0") {
    bucket_counts.resize(1);
    indexes.resize(1);
    bytes = {0u};
    multiproduct_table_kernel<32, 1, 1>
        <<<1, 32>>>(bucket_counts.data(), indexes.data(), bytes.data(), bytes.size());
    expected_counts.resize(1);
    expected_indexes.resize(1);
    basdv::synchronize_device();
    REQUIRE(bucket_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }

  SECTION("we handle the case of a single element of 1") {
    bucket_counts.resize(1);
    indexes.resize(1);
    bytes = {1u};
    multiproduct_table_kernel<32, 1, 1>
        <<<1, 32>>>(bucket_counts.data(), indexes.data(), bytes.data(), bytes.size());
    expected_counts = {1u};
    expected_indexes.resize(1);
    basdv::synchronize_device();
    REQUIRE(bucket_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }

  SECTION("we handle the case of two elements of zero") {
    bucket_counts.resize(1);
    indexes.resize(2);
    bytes = {0u, 0u};
    multiproduct_table_kernel<32, 1, 1>
        <<<1, 32>>>(bucket_counts.data(), indexes.data(), bytes.data(), bytes.size());
    expected_counts = {0u};
    expected_indexes.resize(2);
    basdv::synchronize_device();
    REQUIRE(bucket_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }

  SECTION("we handle the case of two elements of zero and one") {
    bucket_counts.resize(1);
    indexes.resize(2);
    bytes = {0u, 1u};
    multiproduct_table_kernel<32, 1, 1>
        <<<1, 32>>>(bucket_counts.data(), indexes.data(), bytes.data(), bytes.size());
    expected_counts = {1u};
    expected_indexes = {1u, 0u};
    basdv::synchronize_device();
    REQUIRE(bucket_counts == expected_counts);
    REQUIRE(indexes == expected_indexes);
  }
}

TEST_CASE("we can fit parameters for a multiproduct kernel") {
  for (unsigned n = 1; n <= max_multiexponentiation_length_v; ++n) {
    fit_multiproduct_table_kernel(
        [&]<unsigned NumThreads, unsigned ItemsPerThread>(
            std::integral_constant<unsigned, NumThreads>,
            std::integral_constant<unsigned, ItemsPerThread>) noexcept {
          REQUIRE(NumThreads * ItemsPerThread >= n);
          REQUIRE(NumThreads * (ItemsPerThread - 1) < n);
        },
        n);
  }
}
