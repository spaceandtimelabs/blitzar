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
#include "sxt/base/device/property.h"

#include <iostream>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("We can fetch a positive number of gpu devices without crashing the execution") {
  REQUIRE(get_num_devices() >= 0);
}

TEST_CASE("we can get the version of the driver running") {
  auto v1 = get_latest_cuda_version_supported_by_driver();
  auto v2 = get_cuda_version();
  REQUIRE(v1 >= v2);
}

TEST_CASE("we can query info about device memory") {
  size_t bytes_free, bytes_total;
  get_device_mem_info(bytes_free, bytes_total);
  REQUIRE(0 < bytes_free);
  REQUIRE(bytes_free <= bytes_total);
  REQUIRE(bytes_total == get_total_device_memory());
}
