#include "sxt/execution/async/coroutine.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/computation_handle.h"
#include "sxt/execution/async/gpu_computation_event.h"
#include "sxt/execution/async/synchronization.h"
#include "sxt/execution/async/test_kernel.h"
#include "sxt/execution/base/stream.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"

using namespace sxt;
using namespace sxt::xena;

static future<> f_v();
static future<int> f_i();
static future<int> f_i2();

static future<memmg::managed_array<uint64_t>>
f_gpu(const memmg::managed_array<uint64_t>& a, const memmg::managed_array<uint64_t>& b) noexcept;

static future<memmg::managed_array<uint64_t>>
f_gpu2(const memmg::managed_array<uint64_t>& a, const memmg::managed_array<uint64_t>& b) noexcept;

TEST_CASE("futures interoperate with coroutines") {
  SECTION("we can handle a void coroutine") {
    auto res = f_v();
    REQUIRE(res.ready());
  }

  SECTION("we can handle an int coroutine") {
    auto res = f_i();
    REQUIRE(res.ready());
    REQUIRE(res.value() == 123);
  }

  SECTION("we handle a chained coroutine") {
    auto res = f_i2();
    REQUIRE(res.ready());
    REQUIRE(res.value() == 124);
  }

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

static future<> f_v() { co_return; }

static future<int> f_i() { co_return 123; }

static future<int> f_i2() {
  auto x = co_await f_i();
  co_return x + 1;
}

static future<memmg::managed_array<uint64_t>>
f_gpu(const memmg::managed_array<uint64_t>& a, const memmg::managed_array<uint64_t>& b) noexcept {
  xenb::stream stream;
  auto n = a.size();
  memmg::managed_array<uint64_t> res(n);
  add_for_testing(res.data(), stream, a.data(), b.data(), static_cast<int>(n));
  return xena::await_and_own_stream(std::move(stream), std::move(res));
}

static future<memmg::managed_array<uint64_t>>
f_gpu2(const memmg::managed_array<uint64_t>& a, const memmg::managed_array<uint64_t>& b) noexcept {
  auto res = co_await f_gpu(a, b);
  for (auto& x : res) {
    ++x;
  }
  co_return res;
}
