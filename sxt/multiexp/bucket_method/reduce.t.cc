#include "sxt/multiexp/bucket_method/reduce.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we reduce bucket sums into a multiexponentiation result") {
  using E = bascrv::element97;

  std::vector<E> res(1);
  std::vector<E> expected(1);
  std::pmr::vector<E> sums{8, memr::get_managed_device_resource()};

  SECTION("we can reduce a single bucket") {
    sums[0] = 33u;
    auto fut = reduce_buckets<E>(res, sums, 1, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {33u};
    REQUIRE(res == expected);
  }

  SECTION("we can reduce a bucket in second position") {
    sums[1] = 33u;
    auto fut = reduce_buckets<E>(res, sums, 1, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {66u};
    REQUIRE(res == expected);
  }
}
