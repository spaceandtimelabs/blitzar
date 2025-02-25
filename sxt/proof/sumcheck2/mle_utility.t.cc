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
#include "sxt/proof/sumcheck2/mle_utility.h"

#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/scalar25/realization/field.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk2;
using s25t::operator""_s25;

using T = s25t::element;

TEST_CASE("we can copy a slice of mles to device memory") {
  std::pmr::vector<s25t::element> mles{memr::get_managed_device_resource()};
  memmg::managed_array<s25t::element> partial_mles{memr::get_managed_device_resource()};

  basdv::stream stream;

  SECTION("we can copy an mle with a single element") {
    mles = {0x123_s25};
    copy_partial_mles<T>(partial_mles, stream, mles, 1, 0, 1);
    basdv::synchronize_stream(stream);
    memmg::managed_array<s25t::element> expected = {0x123_s25};
    REQUIRE(partial_mles == expected);
  }

  SECTION("we can copy a slice of MLEs") {
    mles = {0x1_s25, 0x2_s25, 0x3_s25, 0x4_s25, 0x5_s25, 0x6_s25};
    copy_partial_mles<T>(partial_mles, stream, mles, 3, 0, 1);
    basdv::synchronize_stream(stream);
    memmg::managed_array<s25t::element> expected = {0x1_s25, 0x3_s25, 0x4_s25, 0x6_s25};
    REQUIRE(partial_mles == expected);
  }
}

TEST_CASE("we can copy partially folded MLEs to the host") {
  std::pmr::vector<s25t::element> device_mles{memr::get_managed_device_resource()};
  std::vector<s25t::element> host_mles;

  basdv::stream stream;

  SECTION("we can copy a single element") {
    device_mles = {0x123_s25};
    host_mles.resize(1);
    copy_folded_mles<T>(host_mles, stream, device_mles, 1, 0, 1);
    basdv::synchronize_stream(stream);
    std::vector<s25t::element> expected = {0x123_s25};
    REQUIRE(host_mles == expected);
  }

  SECTION("we can copy partially folded MLEs") {
    device_mles = {0x123_s25, 0x456_s25};
    host_mles.resize(4);
    copy_folded_mles<T>(host_mles, stream, device_mles, 2, 0, 1);
    basdv::synchronize_stream(stream);
    std::vector<s25t::element> expected = {0x123_s25, 0x0_s25, 0x456_s25, 0x0_s25};
    REQUIRE(host_mles == expected);
  }
}

TEST_CASE("we can query the fraction of device memory taken by MLEs") {
  std::vector<s25t::element> mles;

  SECTION("we handle the zero case") { REQUIRE(get_gpu_memory_fraction<T>(mles) == 0.0); }

  SECTION("the fractions doubles if the length of mles doubles") {
    mles.resize(1);
    auto f1 = get_gpu_memory_fraction<T>(mles);
    REQUIRE(f1 > 0);
    mles.resize(2);
    auto f2 = get_gpu_memory_fraction<T>(mles);
    REQUIRE(f2 == Catch::Approx(2 * f1));
  }
}
