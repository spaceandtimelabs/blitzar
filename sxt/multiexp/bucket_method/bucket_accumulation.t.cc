#include "sxt/multiexp/bucket_method/bucket_accumulation.h"

#include "sxt/execution/schedule/scheduler.h"
#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can perform a bucket accumulation pass") {
  using E = bascrv::element97;
  memmg::managed_array<E> bucket_sums(255 * 32);

  SECTION("we handle the empty case") {
    bucket_sums.reset();
    auto fut = accumulate_buckets<E>(bucket_sums, {}, {});    
    REQUIRE(fut.ready());
  }

  SECTION("we handle a case with a single zero element") {
    uint8_t scalar[32] = {};
    const uint8_t* scalars[] = {
      scalar
    };
    E generators[] = {7};
    auto fut = accumulate_buckets<E>(bucket_sums, generators, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    for (auto val : bucket_sums) {
      REQUIRE(val == 0);
    }
  }
}
