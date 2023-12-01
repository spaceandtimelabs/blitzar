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
#include "sxt/memory/resource/chained_resource.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/counting_resource.h"

using namespace sxt;
using namespace sxt::memr;

TEST_CASE("we can manage a chain of allocations") {
  counting_resource counter;
  {
    chained_resource r{&counter};
    REQUIRE(counter.bytes_allocated() == 0);

    auto ptr = r.allocate(10);
    (void)ptr;
    REQUIRE(counter.bytes_allocated() == 10);
  }
  REQUIRE(counter.bytes_deallocated() == 10);
}
