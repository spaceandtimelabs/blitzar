/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include "sxt/base/device/split.h"

#include <limits>

#include "sxt/base/device/property.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("we can plan chunking around the amount of available device memory") {
  size_t available_device_memory = 1ull << 20;
  size_t split_factor = 4;

  SECTION("chunk size hits targets if possible") {
    auto low = 1.0 / 16.0;
    auto high = 1.0 / 4.0;
    auto bytes = 2u;
    auto opts = plan_split_impl(bytes, available_device_memory, low, high, split_factor);
    REQUIRE(opts.min_chunk_size * bytes / double(available_device_memory) == low);
    REQUIRE(opts.max_chunk_size * bytes / double(available_device_memory) == high);
    REQUIRE(opts.split_factor == split_factor);
  }

  SECTION("chunk size will never be less than one") {
    auto opts = plan_split_impl(1, available_device_memory, std::numeric_limits<double>::epsilon(),
                                std::numeric_limits<double>::epsilon(), split_factor);
    REQUIRE(opts.min_chunk_size == 1);
    REQUIRE(opts.max_chunk_size == 1);
  }

  SECTION("we can do sensible chunking with defaults") {
    auto opts = plan_split(4u);
    REQUIRE(opts.min_chunk_size < opts.max_chunk_size);
    REQUIRE(opts.split_factor >= basdv::get_num_devices());
  }
}
