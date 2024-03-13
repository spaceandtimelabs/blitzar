#include "sxt/multiexp/pippenger2/reduce.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can reduce products") {
  using E = bascrv::element97;

  std::pmr::vector<E> outputs{memr::get_managed_device_resource()};
  std::pmr::vector<E> products{memr::get_managed_device_resource()};
  basdv::stream stream;

  std::pmr::vector<E> expected;

  SECTION("we handle a single element reduction") {
    outputs.resize(1);
    products.resize(1);
    products[0] = 123u;
    reduce_products<E>(outputs, stream, products);
    basdv::synchronize_stream(stream);
    expected = {123u};
    REQUIRE(outputs == expected);
  }

  SECTION("we handle a reduction with two elements") {
    outputs.resize(1);
    products.resize(2);
    products[0] = 123u;
    products[1] = 456u;
    reduce_products<E>(outputs, stream, products);
    basdv::synchronize_stream(stream);
    expected = {123u + 2u * 456u};
    REQUIRE(outputs == expected);
  }
}
