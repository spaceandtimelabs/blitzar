#include "sxt/base/memory/alloc_utility.h"

#include <memory_resource>
#include <type_traits>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basm;

TEST_CASE("we can allocate PODs") {
  std::pmr::monotonic_buffer_resource alloc;

  SECTION("we can allocate a single value") {
    auto obj = allocate_object<double>(&alloc);
    REQUIRE(std::is_same_v<decltype(obj), double*>);
    *obj = 123.456;
  }

  SECTION("we can allocate an array") {
    auto data = allocate_array<double>(&alloc, 3);
    REQUIRE(std::is_same_v<decltype(data), double*>);
    data[0] = 1.23;
    data[1] = 4.56;
    data[2] = 5.78;
  }
}
