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
#include "sxt/algorithm/iteration/for_each.h"

#include <numeric>

#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::algi;

TEST_CASE("we can launch a for_each kernel to iterate over a sequence of integers") {
  for (unsigned n : {0, 1, 2, 3, 5, 31, 32, 33, 63, 64, 65, 100, 1'000, 10'000, 100'000}) {
    memmg::managed_array<unsigned> a{n, memr::get_managed_device_resource()};
    auto data = a.data();
    auto f = [data] __device__ __host__(unsigned /*n*/, unsigned i) noexcept { data[i] = i; };
    auto fut = for_each(f, n);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    memmg::managed_array<unsigned> expected(n);
    std::iota(expected.begin(), expected.end(), 0);
    REQUIRE(a == expected);
  }
}
