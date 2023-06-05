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
#include "sxt/execution/device/computation_handle.h"

#include <random>
#include <utility>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/device/stream_handle.h"
#include "sxt/base/device/stream_pool.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/device/test_kernel.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"

using namespace sxt;
using namespace sxt::xendv;

static memmg::managed_array<uint64_t> make_random_array(std::mt19937& rng, size_t n) noexcept {
  memmg::managed_array<uint64_t> res{n, memr::get_pinned_resource()};
  for (size_t i = 0; i < n; ++i) {
    res[i] = rng();
  }
  return res;
}

TEST_CASE("computation_handle manages a collection of streams") {
  std::mt19937 rng;
  auto pool = basdv::get_stream_pool();

  SECTION("upon destruction, streams are recycled back to the thread_local pool") {
    basdv::stream s;
    auto ptr = s.raw_stream();
    {
      computation_handle comp;
      comp.add_stream(std::move(s));
    }
    auto handle = pool->aquire_handle();
    REQUIRE(handle->stream == ptr);
    pool->release_handle(handle);
  }

  SECTION("we can move construct a computation handle") {
    basdv::stream s;
    computation_handle h1;
    h1.add_stream(std::move(s));

    computation_handle h2{std::move(h1)};
    REQUIRE(h1.empty());
    REQUIRE(!h2.empty());
  }

  SECTION("we can move assign a computation handle") {
    computation_handle h1;
    h1.add_stream(basdv::stream{});

    computation_handle h2;
    h2.add_stream(basdv::stream{});

    h2 = std::move(h1);
    REQUIRE(h1.empty());
    REQUIRE(!h2.empty());
  }

  SECTION("we can wait on an async computation") {
    computation_handle handle;
    size_t n = 10;

    basdv::stream s;
    auto a = make_random_array(rng, n);
    auto b = make_random_array(rng, n);

    memmg::managed_array<uint64_t> c{n, memr::get_pinned_resource()};
    memmg::managed_array<uint64_t> c_dev{n, memr::get_device_resource()};
    add_for_testing(c_dev.data(), s, a.data(), b.data(), n);
    basdv::async_copy_device_to_host(c, c_dev, s.raw_stream());
    handle.add_stream(std::move(s));

    handle.wait();
    REQUIRE(c[0] == a[0] + b[0]);
    REQUIRE(c[n - 1] == a[n - 1] + b[n - 1]);
  }

  SECTION("we can wait on multiple async computations") {
    computation_handle handle;
    size_t n = 10;

    // computation 1
    basdv::stream stream1;
    auto a1 = make_random_array(rng, n);
    auto b1 = make_random_array(rng, n);
    memmg::managed_array<uint64_t> c1{n, memr::get_pinned_resource()};
    memmg::managed_array<uint64_t> c1_dev{n, memr::get_device_resource()};
    add_for_testing(c1_dev.data(), stream1, a1.data(), b1.data(), n);
    basdv::async_copy_device_to_host(c1, c1_dev, stream1.raw_stream());
    handle.add_stream(std::move(stream1));

    // computation 2
    basdv::stream stream2;
    auto a2 = make_random_array(rng, n);
    memmg::managed_array<uint64_t> a2_dev{n, memr::get_device_resource()};
    auto b2 = make_random_array(rng, n);
    memmg::managed_array<uint64_t> c2{n, memr::get_pinned_resource()};
    memmg::managed_array<uint64_t> c2_dev{n, memr::get_device_resource()};
    add_for_testing(c2_dev.data(), stream2, a2.data(), b2.data(), n);
    basdv::async_copy_device_to_host(c2, c2_dev, stream2.raw_stream());
    handle.add_stream(std::move(stream2));

    // wait
    handle.wait();
    REQUIRE(c1[0] == a1[0] + b1[0]);
    REQUIRE(c1[n - 1] == a1[n - 1] + b1[n - 1]);
    REQUIRE(c2[0] == a2[0] + b2[0]);
    REQUIRE(c2[n - 1] == a2[n - 1] + b2[n - 1]);
  }
}
