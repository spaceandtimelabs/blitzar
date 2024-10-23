/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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
#include "sxt/proof/sumcheck/transcript_utility.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

TEST_CASE("we can perform basic operations on a transcript") {
  prft::transcript transcript{"abc"};
  std::vector<s25t::element> p = {0x123_s25};
  s25t::element r, rp;

  SECTION("we don't draw the same challenge from a transcript") {
    round_challenge(r, transcript, p);
    round_challenge(rp, transcript, p);
    REQUIRE(r != rp);

    prft::transcript transcript_p{"abc"};
    p[0] = 0x456_s25;
    round_challenge(rp, transcript_p, p);
    REQUIRE(r != rp);
  }
}
