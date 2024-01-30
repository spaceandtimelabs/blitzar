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

  SECTION("we can accumulate a single element of one") {
    generators = {33u};
    std::vector<uint8_t> scalars1(32);
    scalars1[0] = 1u;
    scalars = {scalars1.data()};
    auto fut = compute_bucket_sums<E>(sums, generators, scalars, element_num_bytes, bit_width);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected[0] = 33u;
    REQUIRE(sums == expected);
  }

  SECTION("we can accumulate a single element of two") {
    generators = {33u};
    std::vector<uint8_t> scalars1(32);
    scalars1[0] = 2u;
    scalars = {scalars1.data()};
    auto fut = compute_bucket_sums<E>(sums, generators, scalars, element_num_bytes, bit_width);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected[1] = 33u;
    REQUIRE(sums == expected);
  }

  SECTION("we can accumulate a multiple elements in the same bucket") {
    generators = {33u, 44u};
    std::vector<uint8_t> scalars1(64);
    scalars1[0] = 1u;
    scalars1[32] = 1u;
    scalars = {scalars1.data()};
    auto fut = compute_bucket_sums<E>(sums, generators, scalars, element_num_bytes, bit_width, 1u);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected[0] = 77u;
    REQUIRE(sums == expected);
  }
}
