#include "sxt/multiexp/curve21/pippenger_multiproduct_solver.h"

#include "sxt/base/container/blob_array.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/overload.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/ristretto/type/literal.h"

using namespace sxt;
using namespace sxt::mtxc21;
using sxt::rstt::operator""_rs;

TEST_CASE("we can use pippenger's algorithm to solve the multiproduct subproblem in "
          "multiexponentiation") {
  pippenger_multiproduct_solver solver;

  SECTION("we handle the empty case") {
    mtxi::index_table products;
    auto res = solver.solve(std::move(products), {}, {}, 0).await_result();
    memmg::managed_array<c21t::element_p3> expected;
    REQUIRE(res == expected);
  }

  SECTION("we handle non-empty products") {
    mtxi::index_table products{{0, 0, 0}, {1, 0, 1, 2}};
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
        0x345_rs,
        0x567_rs,
    };
    basct::blob_array mask(3, 1);
    mask[0][0] = 1;
    mask[1][0] = 1;
    mask[2][0] = 1;
    auto res = solver.solve(std::move(products), generators, mask, 3).await_result();
    REQUIRE(res[0] == generators[0]);
    REQUIRE(res[1] == generators[1] + generators[2]);
  }

  SECTION("we handle generators that aren't used") {
    mtxi::index_table products{{0, 0, 0}, {1, 0, 0, 1}};
    memmg::managed_array<c21t::element_p3> generators = {
        0x123_rs,
        0x345_rs,
        0x567_rs,
    };
    basct::blob_array mask(3, 1);
    mask[0][0] = 1;
    mask[1][0] = 0;
    mask[2][0] = 1;
    auto res = solver.solve(std::move(products), generators, mask, 2).await_result();
    REQUIRE(res[0] == generators[0]);
    REQUIRE(res[1] == generators[0] + generators[2]);
  }
}
