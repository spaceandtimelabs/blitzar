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
#include "sxt/multiexp/curve/multiexponentiation_cpu_driver.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve32/operation/add.h"
#include "sxt/curve32/operation/double.h"
#include "sxt/curve32/operation/neg.h"
#include "sxt/curve32/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/curve/naive_multiproduct_solver.h"
#include "sxt/multiexp/pippenger/multiexponentiation.h"
#include "sxt/multiexp/test/multiexponentiation.h"

using namespace sxt;
using namespace sxt::mtxcrv;

TEST_CASE("we can compute multiexponentiations") {
  naive_multiproduct_solver<c32t::element_p3> solver;
  multiexponentiation_cpu_driver<c32t::element_p3> drv{&solver};
  auto f = [&](basct::cspan<c32t::element_p3> generators,
               basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
    return mtxpi::compute_multiexponentiation(drv, generators, exponents)
        .value()
        .as_array<c32t::element_p3>();
  };
  std::mt19937 rng{9873324};
  mtxtst::exercise_multiexponentiation_fn(rng, f);
}
