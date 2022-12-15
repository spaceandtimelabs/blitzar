#include "sxt/base/device/property.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("We can fetch a positive number of gpu devices without crashing the execution") {
  REQUIRE(get_num_devices() >= 0);
}
