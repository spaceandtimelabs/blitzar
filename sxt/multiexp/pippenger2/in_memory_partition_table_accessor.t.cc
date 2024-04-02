#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor.h"

#include <print>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/temp_file.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can provide access to precomputed partition sums stored on disk") {
  bastst::temp_file temp_file{std::ios::binary};

  using E = bascrv::element97;

  SECTION("we can access a single element") {
    E e{11};
    temp_file.stream().write(reinterpret_cast<const char*>(&e), sizeof(e));
    temp_file.stream().close();

    in_memory_partition_table_accessor<E> accessor{temp_file.name()};
  }
  std::println("{}", temp_file.name());
}
