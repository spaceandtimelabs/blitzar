#include "sxt/proof/sumcheck/proof_computation.h"

#include <utility>
#include <iostream>
#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/proof/sumcheck/cpu_driver.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"
using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

TEST_CASE("we can create a sumcheck proof") {
  prft::transcript transcript{"abc"};
  cpu_driver drv;
  std::vector<s25t::element> polynomials(2);
  std::vector<s25t::element> evaluation_point(1);
  std::vector<s25t::element> mles = {
    0x8_s25,
    0x3_s25,
  };
  std::vector<std::pair<s25t::element, unsigned>> product_table = {
      {0x1_s25, 1},
  };
  std::vector<unsigned> product_terms = {0};
   
  SECTION("we can prove a sum with a single term") {
    auto fut = prove_sum(polynomials, evaluation_point, transcript, drv, mles, product_table,
                         product_terms, 2);
    REQUIRE(polynomials[0] == mles[0]);
    REQUIRE(polynomials[1] == mles[1] - mles[0]);
    std::cout << "*********\n";
    for (auto& r : evaluation_point) {
      std::cout << r << "\n";
    }
    REQUIRE(fut.ready());
  }
}
