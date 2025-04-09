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
#include "sxt/execution/device/to_device_copier.h"

#include <memory_resource>
#include <numeric>
#include <vector>

#include "sxt/base/device/pinned_buffer_pool.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::xendv;

TEST_CASE("we can copy memory from host to device") {
  std::pmr::vector<int> dev{memr::get_managed_device_resource()};
  std::pmr::vector<int> host;
  basdv::stream stream;

  SECTION("we can copy an empty array") {
    to_device_copier copier{dev, stream};
    auto fut = copier.copy(host);
    REQUIRE(fut.ready());
  }

  SECTION("we can a single integer") {
    dev.resize(1);
    to_device_copier copier{dev, stream};

    host = {123};
    auto fut = copier.copy(host);
    REQUIRE(!fut.ready());
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(dev == host);
  }

  SECTION("we can copy two integers in to two passes") {
    dev.resize(2);
    to_device_copier copier{dev, stream};

    host = {123};
    auto fut = copier.copy(host);
    REQUIRE(fut.ready());

    host = {456};
    fut = copier.copy(host);
    REQUIRE(!fut.ready());

    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    std::pmr::vector<int> expected = {123, 456};
    REQUIRE(dev == expected);
  }

  SECTION("we can copy a full buffer") {
    dev.resize(basdv::pinned_buffer_size / sizeof(int));
    to_device_copier copier{dev, stream};
    host.resize(dev.size());
    std::iota(host.begin(), host.end(), 0);
    auto fut = copier.copy(host);
    REQUIRE(!fut.ready());
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(dev == host);
  }

  SECTION("we can copy data larger than the full buffer") {
    dev.resize(basdv::pinned_buffer_size / sizeof(int) + 1u);
    auto sz = dev.size();
    to_device_copier copier{dev, stream};
    host.resize(sz);
    std::iota(host.begin(), host.end(), 0);

    auto fut = copier.copy(basct::cspan<int>{host.data(), sz - 1});
    REQUIRE(fut.ready());

    fut = copier.copy(basct::cspan<int>{host.data() + sz - 1, 1});
    REQUIRE(!fut.ready());

    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(dev == host);
  }

  SECTION("we can copy data larger than two full buffers") {
    dev.resize(2u * basdv::pinned_buffer_size / sizeof(int) + 1u);
    auto sz = dev.size();
    to_device_copier copier{dev, stream};
    host.resize(sz);
    std::iota(host.begin(), host.end(), 0);

    auto fut = copier.copy(basct::cspan<int>{host.data(), sz - 1});
    REQUIRE(!fut.ready());
    xens::get_scheduler().run();

    fut = copier.copy(basct::cspan<int>{host.data() + sz - 1, 1});
    REQUIRE(!fut.ready());

    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(dev == host);
  }

  SECTION("we can copy data many multiples of a single buffer size") {
    dev.resize(10u * basdv::pinned_buffer_size / sizeof(int) + 11u);
    auto sz = dev.size();
    to_device_copier copier{dev, stream};
    host.resize(sz);
    std::iota(host.begin(), host.end(), 0);

    auto fut = copier.copy(host);
    REQUIRE(!fut.ready());
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    REQUIRE(dev == host);
  }
}
