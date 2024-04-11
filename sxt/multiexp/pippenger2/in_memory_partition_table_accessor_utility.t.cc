#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor_utility.h"

#include <random>
#include <vector>

#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can create a partition table accessor from given generators") {
  using E = bascrv::element97;
  std::vector<E> generators(100);
  std::mt19937 rng{0};
  for(auto& g : generators) {
    g = std::uniform_int_distribution<unsigned>{0, 96}(rng);
  }

  SECTION("we can create an accessor from a single generators") {
    auto accessor = make_in_memory_partition_table_accessor<E>(basct::subspan(generators, 0, 1));
    (void)accessor;
  }
}
