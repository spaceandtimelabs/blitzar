#include "sxt/multiexp/bucket_method/sum2.h"

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxbk;

using E = bascrv::element97;

TEST_CASE("we can compute bucket sums") {
  std::vector<E> sums;

  basdv::stream stream;
  std::vector<E> generators;
  std::vector<const uint8_t*> scalars(1);

  unsigned element_num_bytes = 1u;
  unsigned bit_width = 2u;

  SECTION("we handle the empty case") {
    compute_bucket_sums<E>(sums, stream, generators, scalars, element_num_bytes, bit_width);
  }

  SECTION("we handle a sum with a single element and a scalar of zero") {
    sums.resize(12);
    generators = {33u};
    std::vector<uint8_t> scalars1 = {0u};
    scalars = {scalars1.data()};
    compute_bucket_sums<E>(sums, stream, generators, scalars, element_num_bytes, bit_width);
    basdv::synchronize_stream(stream);
    std::vector<E> expected(sums.size(), 0u);
    REQUIRE(sums == expected);
  }

  SECTION("we handle a sum with a single element and a scalar of one") {
    sums.resize(12);
    generators = {33u};
    std::vector<uint8_t> scalars1 = {1u};
    scalars = {scalars1.data()};
    compute_bucket_sums<E>(sums, stream, generators, scalars, element_num_bytes, bit_width);
    basdv::synchronize_stream(stream);
    std::vector<E> expected(sums.size(), 0u);
    expected[0] = 33u;
    REQUIRE(sums == expected);
  }
}
