/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
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
#include "sxt/multiexp/base/generator_utility.h"

#include <vector>

#include "sxt/base/container/blob_array.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::mtxb;

TEST_CASE("we can copy over generators using a mask") {
  SECTION("we handle the case of all generators being copied") {
    std::vector<int> v = {1, 2, 3};
    std::vector<int> res(v.size());
    basct::blob_array masks(v.size(), 1);
    masks[0][0] = 1;
    masks[1][0] = 2;
    masks[2][0] = 3;
    filter_generators<int>(res, v, masks);
    REQUIRE(res == v);
  }

  SECTION("we don't copy elements where the mask is zero") {
    std::vector<int> v = {1, 2, 3};
    std::vector<int> res(2);
    basct::blob_array masks(v.size(), 1);
    masks[0][0] = 1;
    masks[1][0] = 0;
    masks[2][0] = 1;
    filter_generators<int>(res, v, masks);
    std::vector<int> expected = {1, 3};
    REQUIRE(res == expected);
  }
}
