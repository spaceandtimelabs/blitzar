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
#include "sxt/proof/sumcheck/sum_gpu.h"

#include <vector>

#include "sxt/base/iterator/split.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/proof/sumcheck/device_cache.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

TEST_CASE("we can sum MLEs") {
  std::vector<std::pair<s25t::element, unsigned>> product_table;
  std::vector<unsigned> product_terms;
  std::vector<s25t::element> mles;
  std::vector<s25t::element> p(2);

  SECTION("we can sum an MLE with a single term and n=1") {
    product_table = {{0x1_s25, 1}};
    product_terms = {0};
    device_cache cache{product_table, product_terms};
    mles = {0x123_s25};
    auto fut = sum_gpu(p, cache, mles, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == -mles[0]);
  }

  SECTION("we can sum an MLE with a single term, n=1, and a non-unity multiplier") {
    product_table = {{0x2_s25, 1}};
    product_terms = {0};
    device_cache cache{product_table, product_terms};
    mles = {0x123_s25};
    auto fut = sum_gpu(p, cache, mles, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == product_table[0].first * mles[0]);
    REQUIRE(p[1] == -product_table[0].first * mles[0]);
  }

  SECTION("we can sum an MLE with a single term and n=2") {
    product_table = {{0x1_s25, 1}};
    product_terms = {0};
    device_cache cache{product_table, product_terms};
    mles = {0x123_s25, 0x456_s25};
    auto fut = sum_gpu(p, cache, mles, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == mles[1] - mles[0]);
  }

  SECTION("we can sum an MLE with multiple terms and n=1") {
    p.resize(3);
    product_table = {{0x1_s25, 2}};
    product_terms = {0, 1};
    device_cache cache{product_table, product_terms};
    mles = {0x123_s25, 0x456_s25};
    auto fut = sum_gpu(p, cache, mles, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0] * mles[1]);
    REQUIRE(p[1] == -mles[0] * mles[1] - mles[1] * mles[0]);
    REQUIRE(p[2] == mles[0] * mles[1]);
  }

  SECTION("we can sum multiple mles") {
    product_table = {
        {0x1_s25, 1},
        {0x1_s25, 1},
    };
    product_terms = {0, 1};
    device_cache cache{product_table, product_terms};
    mles = {0x123_s25, 0x456_s25};
    auto fut = sum_gpu(p, cache, mles, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0] + mles[1]);
    REQUIRE(p[1] == -mles[0] - mles[1]);
  }

  SECTION("we can chunk sums with n=4") {
    product_table = {{0x1_s25, 1}};
    product_terms = {0};
    device_cache cache{product_table, product_terms};
    mles = {0x123_s25, 0x456_s25, 0x789_s25, 0x91011_s25};
    basit::split_options options{
        .min_chunk_size = 1,
        .max_chunk_size = 1,
        .split_factor = 2,
    };
    auto fut = sum_gpu(p, cache, options, mles, 4);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0] + mles[1]);
    REQUIRE(p[1] == (mles[2] - mles[0]) + (mles[3] - mles[1]));
  }

  SECTION("we can chunk sums with n=4") {
    product_table = {{0x1_s25, 1}};
    product_terms = {0};
    device_cache cache{product_table, product_terms};
    mles = {0x2_s25, 0x4_s25, 0x7_s25, 0x9_s25};
    basit::split_options options{
        .min_chunk_size = 16,
        .max_chunk_size = 16,
        .split_factor = 2,
    };
    auto fut = sum_gpu(p, cache, options, mles, 4);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0] + mles[1]);
    REQUIRE(p[1] == (mles[2] - mles[0]) + (mles[3] - mles[1]));
  }
}
