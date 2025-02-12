#include "cbindings/sumcheck.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/proof/sumcheck/reference_transcript.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"
using namespace sxt;
using s25t::operator""_s25;

TEST_CASE("todo") {
  prft::transcript base_transcript{"abc"};
  prfsk::reference_transcript transcript{base_transcript};

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

  auto f = [](s25t::element* r, void* context, const s25t::element* polynomial,
              unsigned polynomial_len) noexcept {
    static_cast<prfsk::reference_transcript*>(context)->round_challenge(
        *r, {polynomial, polynomial_len});
  };
}
#if 0

  SECTION("we can prove a sum with n=1") {
    auto fut = prove_sum(polynomials, evaluation_point, transcript, drv, mles, product_table,
                         product_terms, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(polynomials[0] == mles[0]);
    REQUIRE(polynomials[1] == -mles[0]);
  }
#endif
