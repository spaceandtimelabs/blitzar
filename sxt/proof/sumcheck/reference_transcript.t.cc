#include "sxt/proof/sumcheck/reference_transcript.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;


TEST_CASE("we provide an implementation of sumcheck transcript") {
  prft::transcript base_transcript{"abc"};
  reference_transcript transcript{base_transcript};
  std::vector<s25t::element> p = {0x123_s25};
  s25t::element r, rp;

  SECTION("we don't draw the same challenge from a transcript") {
    transcript.round_challenge(r, p);
    transcript.round_challenge(rp, p);
    REQUIRE(r != rp);

    prft::transcript base_transcript_p{"abc"};
    reference_transcript transcript_p{base_transcript_p};
    p[0] = 0x456_s25;
    transcript_p.round_challenge(rp, p);
    REQUIRE(r != rp);
  }

  SECTION("init_transcript produces different results based on parameters") {
    transcript.init(1, 2);
    transcript.round_challenge(r, p);

    prft::transcript base_transcript_p{"abc"};
    reference_transcript transcript_p{base_transcript_p};
    transcript.init(2, 1);
    transcript.round_challenge(rp, p);

    REQUIRE(r != rp);
  }
}
