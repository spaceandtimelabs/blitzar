#include "sxt/multiexp/pippenger2/partition_table.h"

#include <iostream>
#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can fill the partition table") {
  using E = bascrv::element97;
  std::vector<E> sums(1u << 16);
  std::vector<E> generators = {1u, 2u,  3u,  4u,  5u,  6u,  7u,  8u,
                               9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u};
  compute_partition_values(sums.data(), generators.data());
  std::cout << sums[1] << "\n";
  std::cout << sums[0b11] << "\n";
  std::cout << sums[0b010] << "\n";
  std::cout << sums[0b100] << "\n";
  std::cout << sums[0b110] << "\n";
  std::cout << sums[0b111] << "\n";
}
