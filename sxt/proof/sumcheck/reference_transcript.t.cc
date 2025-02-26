/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sxt/proof/sumcheck/reference_transcript.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/realization/field.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk2;
using sxt::s25t::operator""_s25;

TEST_CASE("we provide an implementation of sumcheck transcript") {
  using T = s25t::element;
  prft::transcript base_transcript{"abc"};
  reference_transcript<T> transcript{base_transcript};
  std::vector<T> p = {0x123_s25};
  s25t::element r, rp;

  SECTION("we don't draw the same challenge from a transcript") {
    transcript.round_challenge(r, p);
    transcript.round_challenge(rp, p);
    REQUIRE(r != rp);

    prft::transcript base_transcript_p{"abc"};
    reference_transcript<T> transcript_p{base_transcript_p};
    p[0] = 0x456_s25;
    transcript_p.round_challenge(rp, p);
    REQUIRE(r != rp);
  }

  SECTION("init_transcript produces different results based on parameters") {
    transcript.init(1, 2);
    transcript.round_challenge(r, p);

    prft::transcript base_transcript_p{"abc"};
    reference_transcript<T> transcript_p{base_transcript_p};
    transcript.init(2, 1);
    transcript.round_challenge(rp, p);

    REQUIRE(r != rp);
  }
}
