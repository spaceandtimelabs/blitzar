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
#include "sxt/proof/inner_product/generator_fold_kernel.h"

#include "sxt/algorithm/iteration/transform.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/device_viewable.h"
#include "sxt/proof/inner_product/generator_fold.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// async_fold_generators
//--------------------------------------------------------------------------------------------------
xena::future<> async_fold_generators(basct::span<c21t::element_p3> g_vector_p,
                                     basct::cspan<c21t::element_p3> g_vector,
                                     basct::cspan<unsigned> decomposition) noexcept {
  auto np = g_vector_p.size();
  SXT_DEBUG_ASSERT(g_vector.size() == 2u * np);
  struct functor {
    const unsigned* decomposition_data;
    unsigned decomposition_size;

    __device__ __host__ void operator()(c21t::element_p3& lhs,
                                        const c21t::element_p3& rhs) const noexcept {
      fold_generators(lhs, basct::cspan<unsigned>{decomposition_data, decomposition_size}, lhs,
                      rhs);
    }
  };
  auto make_f = [&](std::pmr::polymorphic_allocator<> alloc,
                    basdv::stream& /*stream*/) noexcept -> xena::future<functor> {
    auto decomposition_device = co_await xendv::make_active_device_viewable(alloc, decomposition);
    co_return functor{
        .decomposition_data = decomposition_device.data(),
        .decomposition_size = static_cast<unsigned>(decomposition_device.size()),
    };
  };
  // Note: These haven't been informed by much benchmarking. I'm
  // sure there are better values. This is just putting in some
  // ballpark estimates to get started.
  basit::chunk_options chunk_options{
      .min_size = 1ull << 9u,
      .max_size = 1ull << 18u,
  };
  co_await algi::transform(g_vector_p, chunk_options, make_f, g_vector.subspan(0, np),
                           g_vector.subspan(np));
}
} // namespace sxt::prfip
