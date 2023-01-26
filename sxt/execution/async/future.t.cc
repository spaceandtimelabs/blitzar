#include "sxt/execution/async/future.h"

#include <memory>
#include <random>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/computation_handle.h"
#include "sxt/execution/async/test_kernel.h"
#include "sxt/execution/base/stream.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"

using namespace sxt;
using namespace sxt::xena;

static memmg::managed_array<uint64_t> make_random_array(std::mt19937& rng, size_t n) noexcept {
  memmg::managed_array<uint64_t> res{n, memr::get_pinned_resource()};
  for (size_t i = 0; i < n; ++i) {
    res[i] = rng();
  }
  return res;
}

static future<uint64_t> f1(const memmg::managed_array<uint64_t>& a,
                           const memmg::managed_array<uint64_t>& b) noexcept {
  auto n = a.size();
  memmg::managed_array<uint64_t> a_dev{n, memr::get_device_resource()};
  memmg::managed_array<uint64_t> b_dev{n, memr::get_device_resource()};
  auto num_bytes = n * sizeof(uint64_t);
  xenb::stream s;
  basdv::async_memcpy_host_to_device(a_dev.data(), a.data(), num_bytes, s);
  basdv::async_memcpy_host_to_device(b_dev.data(), b.data(), num_bytes, s);
  memmg::managed_array<uint64_t> c_dev{n, memr::get_device_resource()};

  add_for_testing(c_dev.data(), s, a_dev.data(), b_dev.data(), static_cast<int>(n));

  memmg::managed_array<uint64_t> c{n, memr::get_pinned_resource()};
  basdv::async_memcpy_device_to_host(c.data(), c_dev.data(), num_bytes, s);

  computation_handle handle;
  handle.add_stream(std::move(s));

  // clang-format off
  auto completion = [
    a_dev = std::move(a_dev),
    b_dev = std::move(b_dev),
    c_dev = std::move(c_dev),
    c = std::move(c)
  ](uint64_t& res) noexcept {
    // clang-format on
    res = 0;
    for (auto ci : c) {
      res += ci;
    }
  };
  return future<uint64_t>{std::move(handle), std::move(completion)};
}

static future<void> f2(uint64_t& res, const memmg::managed_array<uint64_t>& a,
                       const memmg::managed_array<uint64_t>& b) noexcept {
  auto n = a.size();
  memmg::managed_array<uint64_t> a_dev{n, memr::get_device_resource()};
  memmg::managed_array<uint64_t> b_dev{n, memr::get_device_resource()};
  auto num_bytes = n * sizeof(uint64_t);
  xenb::stream s;
  basdv::async_memcpy_host_to_device(a_dev.data(), a.data(), num_bytes, s);
  basdv::async_memcpy_host_to_device(b_dev.data(), b.data(), num_bytes, s);
  memmg::managed_array<uint64_t> c_dev{n, memr::get_device_resource()};

  add_for_testing(c_dev.data(), s, a_dev.data(), b_dev.data(), static_cast<int>(n));

  memmg::managed_array<uint64_t> c{n, memr::get_pinned_resource()};
  basdv::async_memcpy_device_to_host(c.data(), c_dev.data(), num_bytes, s);

  computation_handle handle;
  handle.add_stream(std::move(s));

  // clang-format off
  auto completion = [
    a_dev = std::move(a_dev),
    b_dev = std::move(b_dev),
    c_dev = std::move(c_dev),
    c = std::move(c),
    &res
  ]() noexcept {
    // clang-format on
    res = 0;
    for (auto ci : c) {
      res += ci;
    }
  };
  return future<void>{std::move(handle), std::move(completion)};
}

TEST_CASE("future manages the result of an asynchronous computation") {
  std::mt19937 rng;
  int n = 5;
  auto a = make_random_array(rng, n);
  auto b = make_random_array(rng, n);

  uint64_t c = 0;
  for (int i = 0; i < n; ++i) {
    c += a[i] + b[i];
  }

  SECTION("future is default constructible") {
    future<void> f1;
    REQUIRE(!f1.available());

    future<int> f2;
    REQUIRE(!f2.available());
  }

  SECTION("we can make ready futures that won't block") {
    auto f1 = make_ready_future();
    REQUIRE(f1.available());
    f1.await_result();

    auto f2 = make_ready_future(123);
    REQUIRE(f2.available());
    REQUIRE(f2.await_result() == 123);

    auto f3 = make_ready_future(std::make_unique<int>(123));
    REQUIRE(f3.available());
    REQUIRE(*f3.await_result() == 123);
  }

  SECTION("we await asynchronous GPU computations") { REQUIRE(f1(a, b).await_result() == c); }

  SECTION("we can await async GPU computations with a void result") {
    uint64_t res;
    auto fut = f2(res, a, b);
    fut.await_result();
    REQUIRE(res == c);
  }

  SECTION("we can move assign futures with async GPU computations") {
    auto a_p = make_random_array(rng, n);
    auto fut = f1(a_p, b);
    fut = f1(a, b);
    REQUIRE(fut.await_result() == c);
  }

  SECTION("we can destroy a future that's kept running") { auto fut = f1(a, b); }
}
