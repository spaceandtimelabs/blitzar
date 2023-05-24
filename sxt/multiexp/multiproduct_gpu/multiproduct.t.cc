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
#include "sxt/multiexp/multiproduct_gpu/multiproduct.h"

#include <numeric>

#include "sxt/algorithm/base/gather_mapper.h"
#include "sxt/algorithm/reduction/test_reducer.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxmpg;

TEST_CASE("we can compute multiproducts using the GPU") {
  basdv::stream stream;

  using Mapper = algb::gather_mapper<uint64_t>;

  size_t max_generators = 10'000;
  memmg::managed_array<uint64_t> generators{max_generators, memr::get_managed_device_resource()};
  memmg::managed_array<unsigned> indexes{memr::get_managed_device_resource()};
  std::iota(generators.begin(), generators.end(), 1);

  SECTION("we can compute a single product with a single element") {
    indexes = {0};
    memmg::managed_array<unsigned> product_sizes = {1};
    memmg::managed_array<uint64_t> res(product_sizes.size());
    auto fut = compute_multiproduct<algr::test_add_reducer, Mapper>(res, stream, generators,
                                                                    indexes, product_sizes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    memmg::managed_array<uint64_t> expected = {1};
    REQUIRE(res == expected);
  }

  SECTION("we can compute a single product with two terms") {
    indexes = {0, 2};
    memmg::managed_array<unsigned> product_sizes = {2};
    memmg::managed_array<uint64_t> res(product_sizes.size());
    auto fut = compute_multiproduct<algr::test_add_reducer, Mapper>(res, stream, generators,
                                                                    indexes, product_sizes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    memmg::managed_array<uint64_t> expected = {
        generators[0] + generators[2],
    };
    REQUIRE(res == expected);
  }

  SECTION("we can compute multiple products") {
    indexes = {1, 0, 2, 0, 1, 2};
    memmg::managed_array<unsigned> product_sizes = {1, 2, 3};
    memmg::managed_array<uint64_t> res(product_sizes.size());
    auto fut = compute_multiproduct<algr::test_add_reducer, Mapper>(res, stream, generators,
                                                                    indexes, product_sizes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    memmg::managed_array<uint64_t> expected = {
        generators[1],
        generators[0] + generators[2],
        generators[0] + generators[1] + generators[2],
    };
    REQUIRE(res == expected);
  }

  SECTION("we can compute products with many terms") {
    unsigned n = 1'000;
    indexes = memmg::managed_array<unsigned>(n);
    std::iota(indexes.begin(), indexes.end(), 0);
    memmg::managed_array<unsigned> product_sizes = {n};
    memmg::managed_array<uint64_t> res(product_sizes.size());
    auto fut = compute_multiproduct<algr::test_add_reducer, Mapper>(res, stream, generators,
                                                                    indexes, product_sizes);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    memmg::managed_array<uint64_t> expected = {
        n * (n + 1) / 2,
    };
    REQUIRE(res == expected);
  }
}
