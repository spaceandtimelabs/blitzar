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
#include "sxt/proof/inner_product/scalar_fold_kernel.h"

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/future_utility.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// fold_scalars_case1
//--------------------------------------------------------------------------------------------------
static xena::future<> fold_scalars_case1(basct::span<s25t::element> scalars,
                                         const s25t::element& m_low, const s25t::element& m_high,
                                         unsigned mid, unsigned m) noexcept {
  auto data = scalars.data();
  // clang-format off
  auto f = [
    data, 
    m_low, 
    m_high, 
    mid
  ] __device__ __host__(unsigned /*m*/, unsigned i) noexcept 
  {
    auto& x = data[i];
    s25o::mul(x, m_low, x);
    s25o::muladd(x, m_high, data[mid + i], x);
  };
  // clang-format on
  return algi::for_each(f, m);
}

//--------------------------------------------------------------------------------------------------
// fold_scalars_case2
//--------------------------------------------------------------------------------------------------
static xena::future<> fold_scalars_case2(basct::span<s25t::element> scalars,
                                         const s25t::element& m_low, unsigned mid,
                                         unsigned m) noexcept {
  auto data = scalars.data() + m;
  // clang-format off
  auto f = [
    data, 
    m_low
  ] __device__ __host__(unsigned /*m*/, unsigned i) noexcept 
  {
    auto& x = data[i];
    s25o::mul(x, m_low, x);
  };
  // clang-format on
  return algi::for_each(f, mid - m);
}

//--------------------------------------------------------------------------------------------------
// fold_scalars
//--------------------------------------------------------------------------------------------------
xena::future<> fold_scalars(basct::span<s25t::element> scalars, const s25t::element& m_low,
                            const s25t::element& m_high, unsigned mid) noexcept {
  auto n = scalars.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_device_pointer(scalars.data()) &&
      0 < mid && 
      mid < n &&
      n <= 2u * mid
      // clang-format on
  );
  auto m = n - mid;
  auto fut1 = fold_scalars_case1(scalars, m_low, m_high, mid, m);
  if (m == mid) {
    return fut1;
  }
  auto fut2 = fold_scalars_case2(scalars, m_low, mid, m);
  return xena::await_all(std::move(fut1), std::move(fut2));
}
} // namespace sxt::prfip
