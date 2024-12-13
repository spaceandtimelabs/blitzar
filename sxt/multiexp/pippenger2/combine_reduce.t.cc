#include "sxt/multiexp/pippenger2/combine_reduce.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can combine and reduce partial products") {
  using E = bascrv::element97;

  std::vector<unsigned> output_bit_table;
  std::vector<E> partial_products;
  std::vector<E> res(1);

  SECTION("we can combine and reduce a single element") {
    output_bit_table = {1};
    partial_products = {3u};
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u);
  }

  SECTION("we can combine and reduce elements already on device") {
    output_bit_table = {1};
    partial_products = {3u};
    std::pmr::vector<E> partial_products_dev{partial_products.begin(), partial_products.end(),
                                             memr::get_managed_device_resource()};
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products_dev);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u);
  }

  SECTION("we can combine and reduce a single output with a reduction size of two") {
    output_bit_table = {1};
    partial_products = {3u, 4u};
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 7u);
  }

  SECTION("we can combine and reduce an output with a bit width of 2") {
    output_bit_table = {2};
    partial_products = {3u, 4u};
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 11u);
  }

  SECTION("we can combine and reduce multiple outputs") {
    output_bit_table = {1, 1};
    partial_products = {3u, 4u};
    res.resize(2);
    auto fut = combine_reduce<E>(res, output_bit_table, partial_products);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res[0] == 3u);
    REQUIRE(res[1] == 4u);
  }
}
