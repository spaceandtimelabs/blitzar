#include "sxt/proof/sumcheck/verification.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"
using namespace sxt;
using namespace sxt::prfsk;
using sxt::s25t::operator""_s25;

TEST_CASE("we can verify a sumcheck proof up to the polynomial evaluation") {
  s25t::element expected_sum = 0x0_s25;
  std::vector<s25t::element> evaluation_point(1);
  prft::transcript transcript{"abc"};
  std::vector<s25t::element> round_polynomials(1);
}
/* bool verify_sumcheck_no_evaluation(s25t::element& expected_sum, */
/*                                    basct::span<s25t::element> evaluation_point, */
/*                                    prft::transcript& transcript,  */
/*                                    basct::cspan<s25t::element> round_polynomials, */
/*                                    unsigned round_degree) noexcept; */
