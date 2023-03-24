#include "sxt/algorithm/iteration/for_each.h"

#include <numeric>

#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::algi;

TEST_CASE("we can launch a for_each kernel to iterate over a sequence of integers") {
  for (unsigned n : {1, 2, 3, 5, 31, 32, 33, 63, 64, 65, 100, 1'000, 10'000, 100'000}) {
    memmg::managed_array<unsigned> a{n, memr::get_managed_device_resource()};
    auto data = a.data();
    auto f = [data] __device__ __host__(unsigned /*n*/, unsigned i) noexcept { data[i] = i; };
    auto fut = for_each(f, n);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    memmg::managed_array<unsigned> expected(n);
    std::iota(expected.begin(), expected.end(), 0);
    REQUIRE(a == expected);
  }
}
