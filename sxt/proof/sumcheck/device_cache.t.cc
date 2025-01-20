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
#include "sxt/proof/sumcheck/device_cache.h"

#include <vector>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

TEST_CASE("we can cache device values that don't change as a proof is computed") {
  std::vector<std::pair<s25t::element, unsigned>> product_table;
  std::vector<unsigned> product_terms;

  basdv::stream stream;

  basct::cspan<std::pair<s25t::element, unsigned>> product_table_dev;
  basct::cspan<unsigned> product_terms_dev;

  SECTION("we can access values from device memory") {
    product_table = {{0x123_s25, 0}};
    product_terms = {0};
    device_cache cache{product_table, product_terms};
    cache.lookup(product_table_dev, product_terms_dev, stream);

    std::vector<std::pair<s25t::element, unsigned>> product_table_p(product_table.size());
    basdv::async_copy_device_to_host(product_table_p, product_table_dev, stream);

    std::vector<unsigned> product_terms_p(product_terms.size());
    basdv::async_copy_device_to_host(product_terms_p, product_terms_dev, stream);

    basdv::synchronize_stream(stream);
    REQUIRE(product_table_p == product_table);
    REQUIRE(product_terms_p == product_terms);
  }

  SECTION("we can clear the device cache") {
    product_table = {{0x123_s25, 0}};
    product_terms = {0};
    device_cache cache{product_table, product_terms};
    cache.lookup(product_table_dev, product_terms_dev, stream);

    std::vector<std::pair<s25t::element, unsigned>> product_table_p(product_table.size());
    basdv::async_copy_device_to_host(product_table_p, product_table_dev, stream);

    std::vector<unsigned> product_terms_p(product_terms.size());
    basdv::async_copy_device_to_host(product_terms_p, product_terms_dev, stream);

    auto data = cache.clear();
    basdv::async_copy_device_to_host(product_table_p, data->product_table, stream);
    basdv::async_copy_device_to_host(product_terms_p, data->product_terms, stream);
    basdv::synchronize_stream(stream);
    REQUIRE(product_table_p == product_table);
    REQUIRE(product_terms_p == product_terms);
  }
}
