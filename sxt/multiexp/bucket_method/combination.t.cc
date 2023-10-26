#include "sxt/multiexp/bucket_method/combination.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can sum up bucket entries") {
  std::vector<bascrv::element97> sums(1);
  std::vector<bascrv::element97> bucket_sums;

  SECTION("we handle a single bucket") {
    bucket_sums = {12u};
    combine_buckets<bascrv::element97>(sums, bucket_sums);
    REQUIRE(sums[0] == 12u);
  }

  SECTION("we handle 2 buckets") {
    bucket_sums = {2u, 3u};
    combine_buckets<bascrv::element97>(sums, bucket_sums);
    REQUIRE(sums[0] == 2u + 2u * 3u);
  }

  SECTION("we handle 3 buckets") {
    bucket_sums = {2u, 3u, 7u};
    combine_buckets<bascrv::element97>(sums, bucket_sums);
    REQUIRE(sums[0] == 2u + 2u * 3u + 3u * 7u);
  }
}
