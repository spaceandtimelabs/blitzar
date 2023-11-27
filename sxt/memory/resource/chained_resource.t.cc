#include "sxt/memory/resource/chained_resource.h"

#include "sxt/memory/resource/counting_resource.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::memr;

TEST_CASE("we can manage a chain of allocations") {
  counting_resource counter; 
  {
    chained_resource r{&counter};
    REQUIRE(counter.bytes_allocated() == 0);

    auto ptr = r.allocate(10);
    (void)ptr;
    REQUIRE(counter.bytes_allocated() == 10);
  }
  REQUIRE(counter.bytes_deallocated() == 10);
}
