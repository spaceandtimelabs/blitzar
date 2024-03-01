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
#include "sxt/algorithm/transform/prefix_sum.h"

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::algtr;

TEST_CASE("we can compute exclusive prefix sums") {
  basdv::stream stream;

  memmg::managed_array<unsigned> in{memr::get_managed_device_resource()};
  memmg::managed_array<unsigned> out{memr::get_managed_device_resource()};

  SECTION("we handle the empty case") { sxt::algtr::exclusive_prefix_sum(out, in, stream); }

  SECTION("we handle the case of a single element") {
    in = {123};
    out.resize(1);
    sxt::algtr::exclusive_prefix_sum(out, in, stream);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected = {0};
    REQUIRE(out == expected);
  }

  SECTION("we handle two elements") {
    in = {123, 456};
    out.resize(2);
    sxt::algtr::exclusive_prefix_sum(out, in, stream);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected = {0, 123};
    REQUIRE(out == expected);
  }

  SECTION("we can also compute a total aggregation") {
    in = {123, 456};
    out.resize(3);
    sxt::algtr::exclusive_prefix_sum(out, in, stream);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected = {0, 123, 123 + 456};
    REQUIRE(out == expected);
  }
}

TEST_CASE("we can compute inclusive prefix sums") {
  basdv::stream stream;

  memmg::managed_array<unsigned> in{memr::get_managed_device_resource()};
  memmg::managed_array<unsigned> out{memr::get_managed_device_resource()};

  SECTION("we handle the empty case") { sxt::algtr::exclusive_prefix_sum(out, in, stream); }

  SECTION("we handle the case of a single element") {
    in = {123};
    out.resize(1);
    sxt::algtr::inclusive_prefix_sum(out, in, stream);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected = {123};
    REQUIRE(out == expected);
  }

  SECTION("we handle two elements") {
    in = {123, 456};
    out.resize(2);
    sxt::algtr::inclusive_prefix_sum(out, in, stream);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected = {123, 123 + 456};
    REQUIRE(out == expected);
  }
}
