#include "sxt/base/container/span_iterator.h"

#include <vector>

#include "sxt/base/test/unit_test.h"

using namespace sxt::basct;

TEST_CASE("we can iterator over spans of data") {
  SECTION("iterate over clumps of 3") {
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    span_iterator<int> it{data.data(), 3};
    span_iterator<int> last{data.data() + data.size(), 3};
    std::vector<span<int>> v(it, last);
    REQUIRE(v.size() == 2);
    REQUIRE(v[0].data() == &data[0]);
    REQUIRE(v[0].size() == 3);
    REQUIRE(v[1].data() == &data[3]);
    REQUIRE(v[1].size() == 3);
  }
}
