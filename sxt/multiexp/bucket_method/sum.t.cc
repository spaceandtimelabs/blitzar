#include "sxt/multiexp/bucket_method/sum.h"

#include <algorithm>
#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can compute bucket sums") {
  using E = bascrv::element97;
  memmg::managed_array<E> sums(255u);
  const auto element_num_bytes = 1u;
  const auto bit_width = 8u;

  std::vector<E> generators;
  std::vector<const uint8_t*> scalars(1);

  memmg::managed_array<E> expected(sums.size());
  std::fill(expected.begin(), expected.end(), E::identity());

#if 0
  SECTION("we handle the case of no elements") {
    auto fut = compute_bucket_sums<E>(sums, generators, scalars, element_num_bytes, bit_width);
    REQUIRE(fut.ready());
  }
#endif

  SECTION("we handle the case of a single element with a zero exponent") {
    generators = {32u};
    std::vector<uint8_t> scalars1 = {0u};
    scalars = {scalars1.data()};
    auto fut = compute_bucket_sums2<E>(sums, generators, scalars, element_num_bytes, bit_width);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(sums == expected);
  }

  SECTION("we handle the case of a single element with an exponent of 1") {
    generators = {32u};
    std::vector<uint8_t> scalars1 = {1u};
    scalars = {scalars1.data()};
    auto fut = compute_bucket_sums<E>(sums, generators, scalars, element_num_bytes, bit_width);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected[0] = generators[0];
    REQUIRE(sums == expected);
  }

  SECTION("we handle multiple elements that map to the same bucket") {
    generators = {32u, 5u};
    std::vector<uint8_t> scalars1 = {1u, 1u};
    scalars = {scalars1.data()};
    auto fut = compute_bucket_sums<E>(sums, generators, scalars, element_num_bytes, bit_width);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected[0] = 32u + 5u;
    REQUIRE(sums == expected);
  }
}
