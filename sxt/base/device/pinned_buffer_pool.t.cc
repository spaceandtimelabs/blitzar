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
#include "sxt/base/device/pinned_buffer_pool.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("we can pool pinned buffers") {
  size_t num_buffers = 3u;
  pinned_buffer_pool pool{num_buffers};
  REQUIRE(pool.size() == num_buffers);

  SECTION("we can acquire and release buffers") {
    auto h = pool.acquire_handle();
    REQUIRE(pool.size() == num_buffers - 1);
    pool.release_handle(h);
    REQUIRE(pool.size() == num_buffers);
  }

  SECTION("we can acquire a handle from an empty pool") {
    pinned_buffer_pool empty_pool{0};
    auto h = empty_pool.acquire_handle();
    empty_pool.release_handle(h);
    REQUIRE(empty_pool.size() == 1);
  }
}
