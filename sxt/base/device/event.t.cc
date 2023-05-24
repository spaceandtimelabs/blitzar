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
#include "sxt/base/device/event.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("basdv::event provides an RAII wrapper around CUDA events") {
  event ev;

  SECTION("we can move-construct events") {
    bast::raw_cuda_event_t ptr = ev;
    event ev2{std::move(ev)};
    REQUIRE(ev2 == ptr);
    REQUIRE(static_cast<bast::raw_cuda_event_t>(ev) == nullptr);
  }

  SECTION("we can move-assign an event") {
    bast::raw_cuda_event_t ptr = ev;
    event ev2;
    ev2 = std::move(ev);
    REQUIRE(ev2 == ptr);
    REQUIRE(static_cast<bast::raw_cuda_event_t>(ev) == nullptr);
  }
}
