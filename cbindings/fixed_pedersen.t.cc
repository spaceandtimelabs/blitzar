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
#include "cbindings/fixed_pedersen.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/operation/overload.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve21/type/literal.h"

using namespace sxt;
using sxt::c21t::operator""_c21;

struct wrapped_handle {
  wrapped_handle(const c21t::element_p3* generators, unsigned n) noexcept {
    h = sxt_multiexp_handle_new(SXT_CURVE_RISTRETTO255, static_cast<const void*>(generators), n);
  }

  ~wrapped_handle() noexcept { sxt_multiexp_handle_free(h); }

  sxt_multiexp_handle* h;
};

TEST_CASE("we can compute multi-exponentiations with a fixed set of generators") {
  std::vector<c21t::element_p3> generators = {
      0x123_c21,
      0x456_c21,
  };

  const sxt_config config = {SXT_GPU_BACKEND, 0};
  REQUIRE(sxt_init(&config) == 0);

  wrapped_handle h{generators.data(), 2};
  REQUIRE(h.h != nullptr);

  SECTION("we can compute a multiexponentiation") {
    uint8_t scalars[] = {1, 0, 0, 2};
    c21t::element_p3 res;
    sxt_fixed_multiexponentiation(&res, h.h, 2, 1, 2, scalars);
    REQUIRE(res == generators[0] + 2 * 256 * generators[1]);
  }
}
