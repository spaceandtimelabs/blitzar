#include "sxt/multiexp/bucket_method/sum.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can compute the bucket sums for a chunk") {
  using E = bascrv::element97;
  const unsigned element_num_bytes = 32;
  const unsigned bit_width = 8;

  std::vector<E> sums(255 * 32);
  std::vector<E> generators = {3u};
  std::vector<const uint8_t*> scalars;

  std::vector<E> expected(sums.size());

  SECTION("we can compute bucket sums for a single exponent of zero") {
    std::vector<uint8_t> scalars1(32);
    scalars = {scalars1.data()};
    auto fut =
        sum_buckets_chunk<E>(sums, generators, scalars, element_num_bytes, bit_width);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(sums == expected);
  }

  SECTION("we can compute the bucket sums for a single exponent of one") {
    std::vector<uint8_t> scalars1(32);
    scalars1[0] = 1;
    scalars = {scalars1.data()};
    auto fut =
        sum_buckets_chunk<E>(sums, generators, scalars, element_num_bytes, bit_width);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected[0] = generators[0];
    REQUIRE(sums == expected);
  }
}
