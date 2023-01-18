#include "sxt/execution/async/computation_handle.h"

#include <random>
#include <utility>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/test_kernel.h"
#include "sxt/execution/base/stream.h"
#include "sxt/execution/base/stream_handle.h"
#include "sxt/execution/base/stream_pool.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"

using namespace sxt;
using namespace sxt::xena;

memmg::managed_array<double> make_random_array(std::mt19937& rng, size_t n) noexcept {
  memmg::managed_array<double> res{n, memr::get_pinned_resource()};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  for (size_t i = 0; i < n; ++i) {
    res[i] = dist(rng);
  }
  return res;
}

TEST_CASE("computation_handle manages a collection of streams") {
  if (basdv::get_num_devices() == 0) {
    return;
  }

  std::mt19937 rng;
  auto pool = xenb::get_stream_pool();

  SECTION("upon destruction, streams are recycled back to the thread_local pool") {
    xenb::stream s;
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
    xenb::stream s;
    computation_handle h1;
    h1.add_stream(std::move(s));

    computation_handle h2{std::move(h1)};
    REQUIRE(h1.empty());
    REQUIRE(!h2.empty());
  }

  SECTION("we can move assign a computation handle") {
    computation_handle h1;
    h1.add_stream(xenb::stream{});

    computation_handle h2;
    h2.add_stream(xenb::stream{});

    h2 = std::move(h1);
    REQUIRE(h1.empty());
    REQUIRE(!h2.empty());
  }

  SECTION("we can wait on an async computation") {
    computation_handle handle;
    size_t n = 10;
    auto num_bytes = n * sizeof(double);

    xenb::stream s;
    auto a = make_random_array(rng, n);
    memmg::managed_array<double> a_dev{n, memr::get_device_resource()};
    basdv::async_memcpy_host_to_device(a_dev.data(), a.data(), num_bytes, s.raw_stream());

    auto b = make_random_array(rng, n);
    memmg::managed_array<double> b_dev{n, memr::get_device_resource()};
    basdv::async_memcpy_host_to_device(b_dev.data(), b.data(), num_bytes, s.raw_stream());

    memmg::managed_array<double> c{n, memr::get_pinned_resource()};
    memmg::managed_array<double> c_dev{n, memr::get_device_resource()};
    add_for_testing(c_dev.data(), s, a_dev.data(), b_dev.data(), n);
    basdv::async_memcpy_device_to_host(c.data(), c_dev.data(), num_bytes, s.raw_stream());
    handle.add_stream(std::move(s));

    handle.wait();
    REQUIRE(c[0] == a[0] + b[0]);
    REQUIRE(c[n - 1] == a[n - 1] + b[n - 1]);
  }

  SECTION("we can wait on multiple async computations") {
    computation_handle handle;
    size_t n = 10;
    auto num_bytes = n * sizeof(double);

    // computation 1
    xenb::stream stream1;
    auto a1 = make_random_array(rng, n);
    memmg::managed_array<double> a1_dev{n, memr::get_device_resource()};
    basdv::async_memcpy_host_to_device(a1_dev.data(), a1.data(), num_bytes, stream1.raw_stream());
    auto b1 = make_random_array(rng, n);
    memmg::managed_array<double> b1_dev{n, memr::get_device_resource()};
    basdv::async_memcpy_host_to_device(b1_dev.data(), b1.data(), num_bytes, stream1.raw_stream());
    memmg::managed_array<double> c1{n, memr::get_pinned_resource()};
    memmg::managed_array<double> c1_dev{n, memr::get_device_resource()};
    add_for_testing(c1_dev.data(), stream1, a1_dev.data(), b1_dev.data(), n);
    basdv::async_memcpy_device_to_host(c1.data(), c1_dev.data(), num_bytes, stream1.raw_stream());
    handle.add_stream(std::move(stream1));

    // computation 2
    xenb::stream stream2;
    auto a2 = make_random_array(rng, n);
    memmg::managed_array<double> a2_dev{n, memr::get_device_resource()};
    basdv::async_memcpy_host_to_device(a2_dev.data(), a2.data(), num_bytes, stream2.raw_stream());
    auto b2 = make_random_array(rng, n);
    memmg::managed_array<double> b2_dev{n, memr::get_device_resource()};
    basdv::async_memcpy_host_to_device(b2_dev.data(), b2.data(), num_bytes, stream2.raw_stream());
    memmg::managed_array<double> c2{n, memr::get_pinned_resource()};
    memmg::managed_array<double> c2_dev{n, memr::get_device_resource()};
    add_for_testing(c2_dev.data(), stream2, a2_dev.data(), b2_dev.data(), n);
    basdv::async_memcpy_device_to_host(c2.data(), c2_dev.data(), num_bytes, stream2.raw_stream());
    handle.add_stream(std::move(stream2));

    // wait
    handle.wait();
    REQUIRE(c1[0] == a1[0] + b1[0]);
    REQUIRE(c1[n - 1] == a1[n - 1] + b1[n - 1]);
    REQUIRE(c2[0] == a2[0] + b2[0]);
    REQUIRE(c2[n - 1] == a2[n - 1] + b2[n - 1]);
  }
}
