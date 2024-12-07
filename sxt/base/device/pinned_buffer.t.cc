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
#include "sxt/base/device/pinned_buffer.h"

#include "sxt/base/device/pinned_buffer_pool.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("we can manage pinned buffers") {
  auto num_buffers = 5u;
  auto pool = get_pinned_buffer_pool(num_buffers);

  SECTION("we can aquire and release a pinned buffer") {
    {
      pinned_buffer buf;
      REQUIRE(pool->size() == num_buffers - 1);
      *reinterpret_cast<char*>(buf.data()) = 1u;
      *(reinterpret_cast<char*>(buf.data()) + buf.size() - 1) = 2u;
    }
    REQUIRE(pool->size() == num_buffers);
  }

  SECTION("we can move construct a buffer") {
    pinned_buffer buf1;
    auto ptr = buf1.data();
    pinned_buffer buf{std::move(buf1)};
    REQUIRE(buf.data() == ptr);
    REQUIRE(pool->size() == num_buffers - 1);
  }

  SECTION("we can move-assign a buffer") {
    pinned_buffer buf1;
    auto ptr = buf1.data();
    pinned_buffer buf;
    buf = std::move(buf1);
    REQUIRE(buf.data() == ptr);
    REQUIRE(pool->size() == num_buffers - 1);
  }
}
