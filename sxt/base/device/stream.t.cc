/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/base/device/stream.h"

#include "sxt/base/device/property.h"
#include "sxt/base/device/stream_pool.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("stream provides a wrapper around pooled CUDA streams") {
  if (basdv::get_num_devices() == 0) {
    return;
  }

  SECTION("default construction gives us a non-null stream") {
    stream s;
    REQUIRE(s.raw_stream() != nullptr);
  }

  SECTION("we can release a stream") {
    stream s1;
    auto ptr = s1.release_handle();
    REQUIRE(s1.release_handle() == nullptr);
    REQUIRE(ptr != nullptr);
    get_stream_pool()->release_handle(ptr);
  }

  SECTION("we can move construct streams") {
    stream s1;
    auto ptr = s1.raw_stream();
    stream s2{std::move(s1)};
    REQUIRE(s1.release_handle() == nullptr);
    REQUIRE(s2.raw_stream() == ptr);
  }

  SECTION("we can move assign streams") {
    stream s1;
    auto ptr = s1.raw_stream();
    stream s2;
    s2 = std::move(s1);
    REQUIRE(s1.release_handle() == nullptr);
    REQUIRE(s2.raw_stream() == ptr);
  }
}
