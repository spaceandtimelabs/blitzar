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
#include "sxt/proof/inner_product/verification_kernel.h"

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfip;
using sxt::s25t::operator""_s25;

TEST_CASE("we can compute remaining g exponents if initial ones are already computed") {
  memmg::managed_array<s25t::element> g_exponents{memr::get_managed_device_resource()};
  memmg::managed_array<s25t::element> x_sq_vector;

  basdv::stream stream;

  SECTION("we handle the np = 2 case") {
    g_exponents = {0x123_s25, 0x0_s25};
    x_sq_vector = {0x987_s25};
    auto fut = compute_g_exponents_partial(g_exponents, stream, x_sq_vector, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    memmg::managed_array<s25t::element> expected = {
        g_exponents[0],
        g_exponents[0] * x_sq_vector[0],
    };
    REQUIRE(g_exponents == expected);
  }

  SECTION("we handle the np = 4 case") {
    g_exponents = {0x123_s25, 0x0_s25, 0x0_s25, 0x0_s25};
    x_sq_vector = {0x987_s25, 0x623_s25};
    auto fut = compute_g_exponents_partial(g_exponents, stream, x_sq_vector, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    memmg::managed_array<s25t::element> expected = {
        g_exponents[0],
        g_exponents[0] * x_sq_vector[1],
        g_exponents[0] * x_sq_vector[0],
        g_exponents[0] * x_sq_vector[1] * x_sq_vector[0],
    };
    REQUIRE(g_exponents == expected);
  }

  SECTION("we handle round_first > 1") {
    g_exponents = {0x123_s25, 0x876_s25, 0x0_s25, 0x0_s25};
    x_sq_vector = {0x623_s25};
    auto fut = compute_g_exponents_partial(g_exponents, stream, x_sq_vector, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    memmg::managed_array<s25t::element> expected = {
        g_exponents[0],
        g_exponents[1],
        g_exponents[0] * x_sq_vector[0],
        g_exponents[1] * x_sq_vector[0],
    };
    REQUIRE(g_exponents == expected);
  }
}
