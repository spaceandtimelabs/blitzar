#include "sxt/multiexp/pippenger2/combination.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("todo") {
  using E = bascrv::element97;

  std::pmr::vector<E> reduction{1, memr::get_managed_device_resource()};
  std::pmr::vector<E> elements{memr::get_managed_device_resource()};

  std::pmr::vector<E> expected;

  basdv::stream stream;

  SECTION("we can reduce a single element") {
    elements = {123u};
    combine<E>(reduction, stream, elements);
    basdv::synchronize_stream(stream);

    expected = {123u};
    REQUIRE(reduction == expected);
  }
}
