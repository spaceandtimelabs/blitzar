#include "sxt/multiexp/bucket_method/bucket_accumulation.h"

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can perform a bucket accumulation pass") {
  using E = bascrv::element97;
  memmg::managed_array<E> buckets(255 * 32);

  SECTION("OtTH", "we handle the empty case") {
    auto fut = accumulate_buckets<E>(buckets, {}, {});    
    REQUIRE(fut.ready());
  }
}
/* template <bascrv::element T> */
/* xena::future<> accumulate_buckets(basct::span<T> bucket_sums, basct::cspan<T> generators, */
/*                                   basct::cspan<const uint8_t*> exponents) noexcept { */
