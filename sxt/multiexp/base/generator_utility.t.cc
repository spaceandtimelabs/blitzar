#include "sxt/multiexp/base/generator_utility.h"

#include <vector>

#include "sxt/base/container/blob_array.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::mtxb;

TEST_CASE("we can copy over generators using a mask") {
  SECTION("we handle the case of all generators being copied") {
    std::vector<int> v = {1, 2, 3};
    std::vector<int> res(v.size());
    basct::blob_array masks(v.size(), 1);
    masks[0][0] = 1;
    masks[1][0] = 2;
    masks[2][0] = 3;
    filter_generators<int>(res, v, masks);
    REQUIRE(res == v);
  }

  SECTION("we don't copy elements where the mask is zero") {
    std::vector<int> v = {1, 2, 3};
    std::vector<int> res(2);
    basct::blob_array masks(v.size(), 1);
    masks[0][0] = 1;
    masks[1][0] = 0;
    masks[2][0] = 1;
    filter_generators<int>(res, v, masks);
    std::vector<int> expected = {1, 3};
    REQUIRE(res == expected);
  }
}
