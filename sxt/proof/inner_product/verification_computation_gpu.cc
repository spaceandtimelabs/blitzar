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

#include <algorithm>
#include <vector>

#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/proof/inner_product/verification_computation.h"
#include "sxt/proof/inner_product/verification_kernel.h"
#include "sxt/scalar25/operation/inner_product.h"
#include "sxt/scalar25/operation/neg.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// compute_g_exponents_gpu
//--------------------------------------------------------------------------------------------------
static xena::future<> compute_g_exponents_gpu(basct::span<s25t::element> g_exponents,
                                               const s25t::element& allinv,
                                               const s25t::element& ap_value,
                                               basct::cspan<s25t::element> x_sq_vector) noexcept {
  auto num_rounds = x_sq_vector.size();
  auto num_host_rounds = std::min(5ul, x_sq_vector.size());
  auto num_device_rounds = num_rounds - num_host_rounds;
  compute_g_exponents(g_exponents.subspan(0, 1ull << num_host_rounds), allinv, ap_value,
                      x_sq_vector.subspan(num_device_rounds));
  if (num_host_rounds == num_rounds) {
    co_return;
  }
  co_await compute_g_exponents_partial2(g_exponents, x_sq_vector.subspan(0, num_device_rounds + 1),
                                        num_host_rounds);
}

//--------------------------------------------------------------------------------------------------
// compute_g_and_product_exponents
//--------------------------------------------------------------------------------------------------
static xena::future<>
compute_g_and_product_exponents(basct::span<s25t::element> exponents, const s25t::element& allinv,
                                 std::vector<s25t::element> x_sq_vector,
                                 const s25t::element& ap_value,
                                 basct::cspan<s25t::element> b_vector) noexcept {
  auto num_rounds = x_sq_vector.size();
  auto np = 1ull << num_rounds;
  auto g_exponents = exponents.subspan(1, np);
  co_await compute_g_exponents_gpu(g_exponents, allinv, ap_value, x_sq_vector);
  exponents[0] = co_await s25o::async_inner_product(g_exponents, b_vector);
}

//--------------------------------------------------------------------------------------------------
// async_compute_verification_exponents
//--------------------------------------------------------------------------------------------------
xena::future<> async_compute_verification_exponents2(
    basct::span<s25t::element> exponents, basct::cspan<s25t::element> x_vector,
    const s25t::element& ap_value, basct::cspan<s25t::element> b_vector) noexcept {
  auto num_exponents = exponents.size();
  auto num_rounds = x_vector.size();
  auto n = b_vector.size();
  auto np = 1ull << num_rounds;
  // clang-format off
  SXT_DEBUG_ASSERT(
      basdv::is_host_pointer(exponents.data()) &&
      basdv::is_host_pointer(x_vector.data()) &&
      n > 1 &&
      (n == np || n > (1ull << (num_rounds-1))) &&
      num_exponents == 1 + np + 2 * num_rounds &&
      exponents.size() == num_exponents &&
      x_vector.size() == num_rounds &&
      b_vector.size() == n
  );
  // clang-format on
  auto lr_exponents = exponents.subspan(1 + np);
  auto l_exponents = basct::subspan(lr_exponents, 0, num_rounds);
  auto r_exponents = basct::subspan(lr_exponents, num_rounds);
  s25t::element allinv;
  compute_lr_exponents_part1(l_exponents, r_exponents, allinv, x_vector);

  // compute g and product exponents
  co_await compute_g_and_product_exponents(
      exponents.subspan(0, 1 + np), allinv,
      std::vector<s25t::element>{l_exponents.begin(), l_exponents.end()}, ap_value, b_vector);

  // fill in lr exponents
  for (auto& li : l_exponents) {
    s25o::neg(li, li);
  }
}
} // namespace sxt::prfip
