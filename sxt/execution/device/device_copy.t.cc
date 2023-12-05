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
#include "sxt/execution/device/device_copy.h"

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

TEST_CASE("we can copy memory to the active device") {
  memmg::managed_array<int> data_maybe{memr::get_device_resource()};
  memmg::managed_array<int> host_data = {1, 2, 3};
  memr::chained_resource alloc{memr::get_device_resource()};

  SECTION("we handle the empty span") {
    auto fut = winked_device_copy(&alloc, basct::cspan<int>{});
    REQUIRE(!fut.event());
    REQUIRE(fut.value().empty());
  }

  SECTION("host memory is copied to the device") {
    auto fut = winked_device_copy(&alloc, host_data);
    REQUIRE(fut.event());
    xens::get_scheduler().run();
    REQUIRE(basdv::is_active_device_pointer(fut.value().data()));
    REQUIRE(basdv::is_equal_for_testing<int>(fut.value(), host_data));
  }
}
