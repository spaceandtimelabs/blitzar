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
#include "sxt/base/device/pinned_buffer.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("we can manage a buffer of pinned memory") {
  SECTION("we can construct and deconstruct a buffer") {
    pinned_buffer buf;
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.empty());
  }

  SECTION("we can add a single byte to a buffer") {
    pinned_buffer buf;
    std::vector<std::byte> data = {std::byte{123}};
    auto rest = buf.fill_from_host(data);
    REQUIRE(rest.empty());
    REQUIRE(buf.size() == 1);
    REQUIRE(*static_cast<std::byte*>(buf.data()) == data[0]);
  }

  SECTION("we can reset a buffer") {
    pinned_buffer buf;
    std::vector<std::byte> data = {std::byte{123}};
    buf.fill_from_host(data);
    buf.reset();
    REQUIRE(buf.empty());
  }

  SECTION("we can move construct a buffer") {
    pinned_buffer buf;
    std::vector<std::byte> data = {static_cast<std::byte>(123)};
    buf.fill_from_host(data);
    pinned_buffer buf_p{std::move(buf)};
    REQUIRE(buf.empty());
    REQUIRE(buf_p.size() == 1);
    REQUIRE(*static_cast<std::byte*>(buf_p.data()) == data[0]);
  }

  SECTION("we can move assign a buffer") {
    pinned_buffer buf;
    std::vector<std::byte> data = {std::byte{123}};
    buf.fill_from_host(data);

    pinned_buffer buf_p;
    data[0] = std::byte{3};
    buf_p.fill_from_host(data);
    buf_p = std::move(buf);
    REQUIRE(buf.empty());
    REQUIRE(*static_cast<std::byte*>(buf_p.data()) == std::byte{123});
  }

  SECTION("we can fill a buffer") {
    pinned_buffer buf;
    std::vector<std::byte> data(buf.capacity() + 1, std::byte{123});
    auto rest = buf.fill_from_host(data);
    REQUIRE(rest.size() == 1);
    REQUIRE(buf.size() == buf.capacity());
    REQUIRE(buf.full());
  }
}
