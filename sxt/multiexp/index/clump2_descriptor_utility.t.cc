#include "sxt/multiexp/index/clump2_descriptor_utility.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/index/clump2_descriptor.h"
using namespace sxt::mtxi;

TEST_CASE("we can initialize clump2_descriptor") {
  clump2_descriptor descriptor;

  SECTION("we correctly count the number of clump2 subsets") {
    init_clump2_descriptor(descriptor, 2);
    REQUIRE(descriptor.subset_count == 3);

    init_clump2_descriptor(descriptor, 3);
    REQUIRE(descriptor.subset_count == 6);

    init_clump2_descriptor(descriptor, 4);
    REQUIRE(descriptor.subset_count == 10);
  }
}
