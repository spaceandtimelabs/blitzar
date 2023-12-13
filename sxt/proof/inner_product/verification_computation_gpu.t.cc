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
#include "sxt/proof/inner_product/verification_computation_gpu.h"

#include <memory_resource>

#include "sxt/base/device/synchronization.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/proof/inner_product/verification_computation.h"
#include "sxt/scalar25/random/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfip;
using s25t::operator""_s25;

static void baseline_check(basct::cspan<s25t::element> x_vector, const s25t::element& ap_value,
                           basct::cspan<s25t::element> b_vector);

TEST_CASE("exponent computation with the GPU gives the same result as the CPU code") {
  std::pmr::monotonic_buffer_resource alloc;
  basn::fast_random_number_generator rng{1, 2};

  auto ap_value = 0x7682347_s25;

  for (size_t n : {2, 3, 64, 123, 1000}) {
    auto num_rounds = basn::ceil_log2(n);
    memmg::managed_array<s25t::element> x_vector(num_rounds), b_vector(n);
    s25rn::generate_random_elements(x_vector, rng);
    s25rn::generate_random_elements(b_vector, rng);
    baseline_check(x_vector, ap_value, b_vector);
  }
}

static void baseline_check(basct::cspan<s25t::element> x_vector, const s25t::element& ap_value,
                           basct::cspan<s25t::element> b_vector) {
  auto num_rounds = x_vector.size();
  auto np = 1ull << num_rounds;
  auto num_exponents = 1 + np + 2 * num_rounds;
  memmg::managed_array<s25t::element> exponents(num_exponents);
  auto fut = async_compute_verification_exponents(exponents, x_vector, ap_value, b_vector);

  memmg::managed_array<s25t::element> expected(num_exponents);
  compute_verification_exponents(expected, x_vector, ap_value, b_vector);

  xens::get_scheduler().run();
  REQUIRE(fut.ready());

  REQUIRE(exponents == expected);
}
