/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/curve21/base/elligate.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/type/element.h"

using namespace sxt;
using namespace sxt::c21b;

TEST_CASE("we can elligate points") {
  f51t::element r = {
      2017384653874783ull, 2156344215810222ull, 535721083431344ull,
      1371658101679126ull, 272479886743902ull,
  };
  f51t::element x, y;
  int notsquare;
  apply_elligator(x, y, &notsquare, r);

  REQUIRE(notsquare == 0);

  REQUIRE(x[0] == 6774956778639475ull);
  REQUIRE(x[1] == 7163677697396064ull);
  REQUIRE(x[2] == 8128851215186067ull);
  REQUIRE(x[3] == 7440937162974605ull);
  REQUIRE(x[4] == 7128564859470047ull);

  REQUIRE(y[0] == 2248522005865470ull);
  REQUIRE(y[1] == 1916996945195695ull);
  REQUIRE(y[2] == 787068757439750ull);
  REQUIRE(y[3] == 1335669812341514ull);
  REQUIRE(y[4] == 1903247756136345ull);
}
