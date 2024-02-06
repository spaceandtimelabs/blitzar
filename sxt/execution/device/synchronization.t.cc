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
#include "sxt/execution/device/synchronization.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/test_kernel.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"

using namespace sxt;
using namespace sxt::xendv;

static xena::future<memmg::managed_array<uint64_t>>
f_gpu2(const memmg::managed_array<uint64_t>& a, const memmg::managed_array<uint64_t>& b) noexcept;

template <class T>
xena::future<> ft(const T& t) {
  basdv::stream s;
  co_await await_stream(s);
}

TEST_CASE("we can interact with coroutines") {
  SECTION("we handle gpu coroutines") {
    memmg::managed_array<uint64_t> a = {1, 2, 3};
    memmg::managed_array<uint64_t> b = {4, 5, 6};
    auto res = f_gpu2(a, b);
    REQUIRE(!res.ready());
    xens::get_scheduler().run();
    REQUIRE(res.ready());
    memmg::managed_array<uint64_t> expected = {6, 8, 10};
    REQUIRE(res.value() == expected);
  }
}

static xena::future<memmg::managed_array<uint64_t>>
f_gpu(const memmg::managed_array<uint64_t>& a, const memmg::managed_array<uint64_t>& b) noexcept {
  basdv::stream stream;
  auto n = a.size();
  memmg::managed_array<uint64_t> res(n);
  add_for_testing(res.data(), stream, a.data(), b.data(), static_cast<int>(n));
  return await_and_own_stream(std::move(stream), std::move(res));
}

static xena::future<memmg::managed_array<uint64_t>>
f_gpu2(const memmg::managed_array<uint64_t>& a, const memmg::managed_array<uint64_t>& b) noexcept {
  auto res = co_await f_gpu(a, b);
  for (auto& x : res) {
    ++x;
  }
  co_return res;
}
