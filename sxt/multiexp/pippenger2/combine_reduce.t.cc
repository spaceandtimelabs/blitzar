#include "sxt/multiexp/pippenger2/combine_reduce.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
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
/* template <bascrv::element T> */
/* xena::future<> combine_reduce(basct::span<T> res, basct::cspan<unsigned> output_bit_table, */
/*                               basct::cspan<T> partial_products) noexcept { */
}
