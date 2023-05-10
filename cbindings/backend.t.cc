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
#include "cbindings/backend.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::cbn;

TEST_CASE("We can verify that invalid backends will error out") {
  const sxt_config config = {-1, 0};
  REQUIRE(sxt_init(&config) != 0);
  REQUIRE(is_backend_initialized() == false);
}

static void test_backend_initialization(int backend) {
  SECTION("The backend is not initialized before calling `sxt_init` function") {
    REQUIRE(is_backend_initialized() == false);
  }

  SECTION("The backend is initialized after calling `sxt_init` function with zero precomputed "
          "elements") {
    const sxt_config config = {backend, 0};
    REQUIRE(sxt_init(&config) == 0);
    REQUIRE(is_backend_initialized() == true);
    reset_backend_for_testing();
  }

  SECTION("The backend is initialized after calling `sxt_init` function with non-zero precomputed "
          "elements") {
    const sxt_config config = {backend, 10};
    REQUIRE(sxt_init(&config) == 0);
    REQUIRE(is_backend_initialized() == true);
    reset_backend_for_testing();
  }

  SECTION("The backend is not initialized after calling the reset function") {
    const sxt_config config = {backend, 0};
    REQUIRE(sxt_init(&config) == 0);
    reset_backend_for_testing();
    REQUIRE(is_backend_initialized() == false);
  }
}

TEST_CASE("We can correctly initialize the naive gpu backend") {
  test_backend_initialization(SXT_GPU_BACKEND);
}

TEST_CASE("We can correctly initialize the pippenger cpu backend") {
  test_backend_initialization(SXT_CPU_BACKEND);
}
