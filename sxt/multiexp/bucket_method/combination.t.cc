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
#include "sxt/multiexp/bucket_method/combination.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can sum up bucket entries") {
  std::vector<bascrv::element97> sums(1);
  std::vector<bascrv::element97> bucket_sums;

  SECTION("we handle a single bucket") {
    bucket_sums = {12u};
    combine_buckets<bascrv::element97>(sums, bucket_sums);
    REQUIRE(sums[0] == 12u);
  }

  SECTION("we handle 2 buckets") {
    bucket_sums = {2u, 3u};
    combine_buckets<bascrv::element97>(sums, bucket_sums);
    REQUIRE(sums[0] == 2u + 2u * 3u);
  }

  SECTION("we handle 3 buckets") {
    bucket_sums = {2u, 3u, 7u};
    combine_buckets<bascrv::element97>(sums, bucket_sums);
    REQUIRE(sums[0] == 2u + 2u * 3u + 3u * 7u);
  }
}
