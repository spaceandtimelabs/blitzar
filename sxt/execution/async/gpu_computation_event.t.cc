#include "sxt/execution/async/gpu_computation_event.h"

#include <memory>
#include <random>

#include "sxt/base/device/event.h"
#include "sxt/base/device/event_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/computation_handle.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/test_kernel.h"
#include "sxt/execution/base/stream.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"

using namespace sxt;
using namespace sxt::xena;

static memmg::managed_array<uint64_t> make_random_array(std::mt19937& rng, size_t n) noexcept;

static future<memmg::managed_array<uint64_t>> f1(const memmg::managed_array<uint64_t>& a,
                                                 const memmg::managed_array<uint64_t>& b) noexcept;

TEST_CASE("we can create futures out of GPU computations") {
  std::mt19937 rng{0};
  size_t n = 100;
  auto a = make_random_array(rng, n);
  auto b = make_random_array(rng, n);

  memmg::managed_array<uint64_t> expected(n);
  for (size_t i = 0; i < n; ++i) {
    expected[i] = a[i] + b[i];
  }

  SECTION("we can add two arrays together") {
    auto c_fut = f1(a, b);
    REQUIRE(!c_fut.ready());
    xens::get_scheduler().run();
    REQUIRE(c_fut.ready());
    REQUIRE(c_fut.value() == expected);
  }

  SECTION("we can run a future with a continuation") {
    bool flag = false;
    auto c_fut = f1(a, b).then([&](memmg::managed_array<uint64_t>&& res) noexcept {
      flag = true;
      return res;
    });
    REQUIRE(!c_fut.ready());
    REQUIRE(!flag);
    xens::get_scheduler().run();
    REQUIRE(c_fut.ready());
    REQUIRE(c_fut.value() == expected);
  }
}

static memmg::managed_array<uint64_t> make_random_array(std::mt19937& rng, size_t n) noexcept {
  memmg::managed_array<uint64_t> res{n, memr::get_pinned_resource()};
  for (size_t i = 0; i < n; ++i) {
    res[i] = rng();
  }
  return res;
}

static future<memmg::managed_array<uint64_t>> f1(const memmg::managed_array<uint64_t>& a,
                                                 const memmg::managed_array<uint64_t>& b) noexcept {
  auto n = a.size();
  xenb::stream s;
  memmg::managed_array<uint64_t> c{n, memr::get_pinned_resource()};
  add_for_testing(c.data(), s, a.data(), b.data(), static_cast<int>(n));
  basdv::event event;
  basdv::record_event(event, s);
  promise<memmg::managed_array<uint64_t>> p;
  future_state<memmg::managed_array<uint64_t>> state;
  state.emplace(std::move(c));
  xena::future<memmg::managed_array<uint64_t>> res{p, std::move(state)};
  computation_handle handle;
  handle.add_stream(std::move(s));
  auto gpu_event = std::make_unique<gpu_computation_event<memmg::managed_array<uint64_t>>>(
      std::move(event), std::move(handle), std::move(p));
  xens::get_scheduler().schedule(std::move(gpu_event));
  return res;
}
