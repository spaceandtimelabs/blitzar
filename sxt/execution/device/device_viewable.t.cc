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
#include "sxt/execution/device/device_viewable.h"

#include "sxt/base/device/active_device_guard.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/chained_resource.h"
#include "sxt/memory/resource/device_resource.h"

using namespace sxt;
using namespace sxt::xendv;

TEST_CASE("we can make a span of memory viewable to the active device") {
  memmg::managed_array<int> data_maybe{memr::get_device_resource()};
  memmg::managed_array<int> host_data = {1, 2, 3};
  memr::chained_resource alloc{memr::get_device_resource()};

  SECTION("we handle the empty span") {
    auto viewable = make_active_device_viewable(data_maybe, basct::cspan<int>{});
    REQUIRE(!viewable.event());
    REQUIRE(viewable.value().empty());
  }

  SECTION("host memory is copied to the device") {
    auto viewable = make_active_device_viewable(data_maybe, host_data);
    REQUIRE(viewable.event());
    REQUIRE(viewable.value().data() == data_maybe.data());
    REQUIRE(basdv::is_active_device_pointer(data_maybe.data()));
    xens::get_scheduler().run();
    REQUIRE(basdv::is_equal_for_testing<int>(viewable.value(), host_data));
  }

  SECTION("we use winked_allocations") {
    auto viewable = make_active_device_viewable(&alloc, host_data);
    REQUIRE(viewable.event());
    xens::get_scheduler().run();
    REQUIRE(basdv::is_active_device_pointer(viewable.value().data()));
    REQUIRE(basdv::is_equal_for_testing<int>(viewable.value(), host_data));
  }

  SECTION("memory already on the device isn't copied") {
    auto viewable = make_active_device_viewable(data_maybe, host_data);
    xens::get_scheduler().run();
    memmg::managed_array<int> data_maybe_p{memr::get_device_resource()};
    auto viewable_p = make_active_device_viewable(data_maybe_p, viewable.value());
    REQUIRE(!viewable_p.event());
    REQUIRE(viewable_p.value().data() == viewable.value().data());
    REQUIRE(basdv::is_equal_for_testing<int>(viewable_p.value(), host_data));
  }

  SECTION("we can make memory on a different device viewable") {
    auto device_p = basdv::get_num_devices() - 1;
    memmg::managed_array<int> data_maybe_p{memr::get_device_resource()};
    basct::cspan<int> other_data;
    {
      basdv::active_device_guard active_guard{device_p};
      other_data = make_active_device_viewable(data_maybe_p, host_data).value();
    }
    xens::get_scheduler().run();
    auto viewable = make_active_device_viewable(data_maybe, other_data);
    REQUIRE(basdv::is_active_device_pointer(viewable.value().data()));
    xens::get_scheduler().run();
    REQUIRE(basdv::is_equal_for_testing<int>(viewable.value(), host_data));
  }
}
