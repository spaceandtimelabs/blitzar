#include "sxt/execution/async/future_utility.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"

using namespace sxt;
using namespace sxt::xena;

static future<> f(promise<>& p1, promise<>& p2) noexcept;

TEST_CASE("we can await multiple futures") {
  promise<> p1, p2;
  auto fut = f(p1, p2);
  REQUIRE(!fut.ready());
  p1.make_ready();
  REQUIRE(!fut.ready());
  p2.make_ready();
  REQUIRE(fut.ready());
}

static future<> f(promise<>& p1, promise<>& p2) noexcept {
  future<> fut1{p1}, fut2{p2};
  return await_all(std::move(fut1), std::move(fut2));
}
