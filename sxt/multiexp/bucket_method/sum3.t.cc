#include "sxt/multiexp/bucket_method/sum3.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
using namespace sxt;
using namespace sxt::mtxbk;

using E = bascrv::element97;

TEST_CASE("we accumulate buckets") {
  unsigned element_num_bytes = 32;
  unsigned bit_width = 8;
  unsigned num_buckets_per_output = ((1u << bit_width) - 1u) * element_num_bytes;

  std::vector<E> sums;
  std::vector<E> generators;
  std::vector<const uint8_t*> scalars;

  sums.resize(num_buckets_per_output);

  std::vector<E> expected(num_buckets_per_output, 0u);

  SECTION("we can accumulate a single element of zero") {
    generators = {33u};
    std::vector<uint8_t> scalars1(32);
    scalars = {scalars1.data()};
    auto fut = compute_bucket_sums<E>(sums, generators, scalars, element_num_bytes, bit_width);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(sums == expected);
  }
/* template <bascrv::element T> */
/* xena::future<> compute_bucket_sums(basct::span<T> sums, basct::cspan<T> generators, */
/*                                    basct::cspan<const uint8_t*> scalars, unsigned element_num_bytes, */
/*                                    unsigned bit_width) noexcept { */
}
