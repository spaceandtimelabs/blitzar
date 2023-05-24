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
#include "sxt/base/device/stream_pool.h"

#include "sxt/base/device/property.h"
#include "sxt/base/device/stream_handle.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("we can pool cuda streams for reuse") {
  if (basdv::get_num_devices() == 0) {
    return;
  }

  SECTION("if we aquire a stream handle it is non-null") {
    stream_pool pool{1};
    auto handle = pool.aquire_handle();
    REQUIRE(handle != nullptr);
    REQUIRE(handle->stream != nullptr);
    REQUIRE(handle->next == nullptr);
    pool.release_handle(handle);
  }

  SECTION("stream handles are reused") {
    stream_pool pool{1};
    auto handle = pool.aquire_handle();
    pool.release_handle(handle);
    auto handle_p = pool.aquire_handle();
    REQUIRE(handle == handle_p);
    pool.release_handle(handle_p);
  }

  SECTION("new handles are made if none are available") {
    stream_pool pool{0};
    auto handle = pool.aquire_handle();
    REQUIRE(handle != nullptr);
    pool.release_handle(handle);
    auto handle_p = pool.aquire_handle();
    REQUIRE(handle == handle_p);
    pool.release_handle(handle_p);
  }

  SECTION("we can acquire handles from a thread-local pool") {
    auto pool = get_stream_pool();
    auto handle = pool->aquire_handle();
    REQUIRE(handle != nullptr);
    REQUIRE(handle->stream != nullptr);
    REQUIRE(handle->next == nullptr);
    pool->release_handle(handle);
  }
}
