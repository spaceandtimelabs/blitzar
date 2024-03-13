#include "sxt/multiexp/pippenger2/multiexponentiation.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute multiexponentiations") {
  using E = bascrv::element97;

  std::vector<E> res(1);
  std::vector<E> partition_table(1u << 16u);
  std::vector<const uint8_t*> scalars;

  std::vector<E> expected(1);

  SECTION("we handle a single element of zero") {
    std::vector<uint8_t> scalars1(32);
    scalars = {scalars1.data()};
    auto fut = multiexponentiate<E>(res, partition_table, scalars, 32, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(res == expected);
  }

  SECTION("we handle a single element of one") {
    std::vector<uint8_t> scalars1(32);
    scalars1[0] = 1u;
    scalars = {scalars1.data()};
    partition_table[1] = 123u;
    auto fut = multiexponentiate<E>(res, partition_table, scalars, 32, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected = {123u};
    REQUIRE(res == expected);
  }
}
