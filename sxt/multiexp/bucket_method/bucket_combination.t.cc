#include "sxt/multiexp/bucket_method/bucket_combination.h"

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
}
