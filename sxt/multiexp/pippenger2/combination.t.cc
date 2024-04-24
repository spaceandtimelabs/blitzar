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
#include "sxt/multiexp/pippenger2/combination.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("todo") {
  using E = bascrv::element97;

  std::pmr::vector<E> reduction{1, memr::get_managed_device_resource()};
  std::pmr::vector<E> elements{memr::get_managed_device_resource()};

  std::pmr::vector<E> expected;

  basdv::stream stream;

  SECTION("we can reduce a single element") {
    elements = {123u};
    combine<E>(reduction, stream, elements);
    basdv::synchronize_stream(stream);

    expected = {123u};
    REQUIRE(reduction == expected);
  }

  SECTION("we can reduce two elements") {
    elements = {3u, 4u};
    combine<E>(reduction, stream, elements);
    basdv::synchronize_stream(stream);

    expected = {7u};
    REQUIRE(reduction == expected);
  }

  SECTION("we can reduce multiple elements") {
    reduction.resize(3);
    elements = {3u, 4u, 1u, 2u, 6u, 5u};
    combine<E>(reduction, stream, elements);
    basdv::synchronize_stream(stream);

    expected = {3 + 2, 4 + 6, 1 + 5};
    REQUIRE(reduction == expected);
  }
}
