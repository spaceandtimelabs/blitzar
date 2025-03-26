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
#include "sxt/execution/device/for_each.h"

#include <utility>
#include <vector>

#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"

using namespace sxt;
using namespace sxt::xendv;

TEST_CASE("we can concurrently invoke code on different GPUs") {
  std::vector<std::pair<size_t, size_t>> ranges;

  SECTION("we handle the empty case") {
    auto fut =
        concurrent_for_each(basit::index_range{0, 0}, [&](const basit::index_range& rng) noexcept {
          ranges.emplace_back(rng.a(), rng.b());
          return xena::make_ready_future();
        });
    REQUIRE(fut.ready());
    REQUIRE(ranges.empty());
  }

  SECTION("we handle ranges with a single element") {
    auto fut =
        concurrent_for_each(basit::index_range{1, 2}, [&](const basit::index_range& rng) noexcept {
          ranges.emplace_back(rng.a(), rng.b());
          return xena::make_ready_future();
        });
    REQUIRE(fut.ready());
    std::vector<std::pair<size_t, size_t>> expected = {{1, 2}};
    REQUIRE(ranges == expected);
  }

  SECTION("we handle ranges with arbitrary number of elements") {
    auto fut =
        concurrent_for_each(basit::index_range{1, 11}, [&](const basit::index_range& rng) noexcept {
          ranges.emplace_back(rng.a(), rng.b());
          return xena::make_ready_future();
        });
    REQUIRE(fut.ready());
    REQUIRE(!ranges.empty());
    REQUIRE(ranges[0].first == 1);
    size_t t = ranges[0].second;
    for (size_t i = 1; i < ranges.size(); ++i) {
      REQUIRE(ranges[i].first == t);
      t = ranges[i].second;
    }
    REQUIRE(t == 11);
  }
}

TEST_CASE("we can manage asynchronous chunked computations") {
  std::vector<std::pair<unsigned, unsigned>> ranges;
  xena::promise<int> ps;
  
  SECTION("we handle no chunks") {
    basit::index_range_iterator iter{basit::index_range{2, 2}, 1};
    auto fut = for_each_device(
        iter, iter, [&](const device_context& ctx, basit::index_range rng) -> xena::future<> {
        return xena::future<int>{ps}.then([](int /*val*/) noexcept {
        });
    });
    REQUIRE(fut.ready());
  }
}
