#include "sxt/base/type/value_type.h"

#include <vector>
#include <type_traits>

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::bast;

TEST_CASE("we can get the value type of containers") {
  REQUIRE(std::is_same_v<value_type_t<std::vector<int>>, int>);
}
