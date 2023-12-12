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
#include "sxt/algorithm/iteration/transform.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/future_utility.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// async_fold_scalars
//--------------------------------------------------------------------------------------------------
xena::future<> async_fold_scalars(basct::span<s25t::element> scalars_p,
                                  basct::cspan<s25t::element> scalars, const s25t::element& m_low,
                                  const s25t::element& m_high) noexcept {
  auto mid = scalars_p.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      scalars_p.size() == mid &&
      mid < scalars.size() && scalars.size() <= 2u * mid
      // clang-format on
  );
  auto f1 = [m_low, m_high] __device__ __host__(s25t::element & x,
                                                const s25t::element& y) noexcept {
    s25o::mul(x, m_low, x);
    s25o::muladd(x, m_high, y, x);
  };
  auto m = scalars.size() - mid;
  // Note: These haven't been informed by much benchmarking. I'm
  // sure there are better values. This is just putting in some
  // ballpark estimates to get started.
  basit::chunk_options chunk_options{
      .min_size = 2u << 10u,
      .max_size = 2u << 20u,
  };

  // case 1
  auto fut1 = algi::transform(scalars_p.subspan(0, m), chunk_options, f1, scalars.subspan(0, m),
                              scalars.subspan(mid));

  // case 2
  auto f2 = [m_low] __device__ __host__(s25t::element & x) noexcept { s25o::mul(x, m_low, x); };
  co_await algi::transform(scalars_p.subspan(m), chunk_options, f2, scalars.subspan(m, mid - m));

  co_await std::move(fut1);
}
} // namespace sxt::prfip
