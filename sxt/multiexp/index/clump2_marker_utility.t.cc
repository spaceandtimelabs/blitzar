#include "sxt/multiexp/index/clump2_marker_utility.h"

#include <iostream>

#include "sxt/multiexp/index/clump2_descriptor.h"
#include "sxt/multiexp/index/clump2_descriptor_utility.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxi;

TEST_CASE("we can convert between a clumped index set and its marker") {
  clump2_descriptor descriptor;

  uint64_t marker, clump_index, index1, index2;

  SECTION("verify conversions for a clump size of 2") {
    init_clump2_descriptor(descriptor, 2);

    marker = compute_clump2_marker(descriptor, 0, 0, 0);
    REQUIRE(marker == 0);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 0);
    REQUIRE(index2 == 0);

    marker = compute_clump2_marker(descriptor, 0, 0, 1);
    REQUIRE(marker == 1);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 0);
    REQUIRE(index2 == 1);

    marker = compute_clump2_marker(descriptor, 0, 1, 1);
    REQUIRE(marker == 2);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 1);
    REQUIRE(index2 == 1);

    marker = compute_clump2_marker(descriptor, 10, 1, 1);
    REQUIRE(marker == 10 * 3 + 2);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 10);
    REQUIRE(index1 == 1);
    REQUIRE(index2 == 1);
  }

  SECTION("verify conversions for a clump size of 3") {
    init_clump2_descriptor(descriptor, 3);

    marker = compute_clump2_marker(descriptor, 0, 0, 0);
    REQUIRE(marker == 0);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 0);
    REQUIRE(index2 == 0);

    marker = compute_clump2_marker(descriptor, 0, 0, 1);
    REQUIRE(marker == 1);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 0);
    REQUIRE(index2 == 1);

    marker = compute_clump2_marker(descriptor, 0, 0, 2);
    REQUIRE(marker == 2);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 0);
    REQUIRE(index2 == 2);

    marker = compute_clump2_marker(descriptor, 0, 1, 1);
    REQUIRE(marker == 3);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 1);
    REQUIRE(index2 == 1);

    marker = compute_clump2_marker(descriptor, 0, 1, 2);
    REQUIRE(marker == 4);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 1);
    REQUIRE(index2 == 2);

    marker = compute_clump2_marker(descriptor, 0, 2, 2);
    REQUIRE(marker == 5);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 2);
    REQUIRE(index2 == 2);

    std::cout << marker << std::endl;
    std::cout << "clump_index = " << clump_index << "\n";
    std::cout << "index1 = " << index1 << "\n";
    std::cout << "index2 = " << index2 << "\n";
  }
  (void)clump_index;
  (void)index1;
  (void)index2;
}
