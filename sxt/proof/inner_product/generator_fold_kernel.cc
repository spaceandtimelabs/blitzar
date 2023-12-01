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

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/proof/inner_product/generator_fold.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// fold_generators_partial
//--------------------------------------------------------------------------------------------------
static xena::future<> fold_generators_partial(basct::span<c21t::element_p3> g_vector_p,
                                              basct::cspan<c21t::element_p3> g_vector,
                                              basct::cspan<unsigned> decomposition,
                                              basit::index_range rng) noexcept {
  auto n = g_vector_p.size();
  auto partial_size = rng.size();
  basdv::stream stream;
  memr::async_device_resource resource{stream};

  // partial_g_vector
  memmg::managed_array<c21t::element_p3> partial_g_vector{2u * partial_size, &resource};
  basdv::async_copy_host_to_device(basct::subspan(partial_g_vector, 0, partial_size),
                                   g_vector.subspan(rng.a(), partial_size), stream);
  basdv::async_copy_host_to_device(basct::subspan(partial_g_vector, partial_size),
                                   g_vector.subspan(rng.a() + n, partial_size), stream);

  // decomposition_gpu
  memmg::managed_array<unsigned> decomposition_gpu{decomposition.size(), &resource};
  basdv::async_copy_host_to_device(decomposition_gpu, decomposition, stream);
  auto decomposition_data = decomposition_gpu.data();
  auto decomposition_size = static_cast<unsigned>(decomposition.size());

  // launch kernel
  auto data = partial_g_vector.data();
  auto f = [
               // clang-format off
    data,
    decomposition_data,
    decomposition_size
               // clang-format on
  ] __device__
           __host__(unsigned partial_size, unsigned i) noexcept {
             fold_generators(data[i],
                             basct::cspan<unsigned>{decomposition_data, decomposition_size},
                             data[i], data[i + partial_size]);
           };
  algi::launch_for_each_kernel(stream, f, partial_size);

  // copy result
  basdv::async_copy_device_to_host(g_vector_p.subspan(rng.a(), partial_size),
                                   basct::subspan(partial_g_vector, 0, partial_size), stream);
  return xendv::await_and_own_stream(std::move(stream));
}

//--------------------------------------------------------------------------------------------------
// fold_generators_impl
//--------------------------------------------------------------------------------------------------
xena::future<void> fold_generators_impl(basct::span<c21t::element_p3> g_vector_p,
                                        basct::cspan<c21t::element_p3> g_vector,
                                        basct::cspan<unsigned> decomposition, size_t split_factor,
                                        size_t min_chunk_size, size_t max_chunk_size) noexcept {
  auto n = g_vector_p.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      n > 0 &&
      g_vector_p.size() == n &&
      g_vector.size() == 2u * n &&
      basdv::is_host_pointer(g_vector_p.data()) &&
      basdv::is_host_pointer(g_vector.data()) &&
      basdv::is_host_pointer(decomposition.data())
      // clang-format on
  );

  auto [first, last] = basit::split(
      basit::index_range{0, n}.min_chunk_size(min_chunk_size).max_chunk_size(max_chunk_size),
      split_factor);

  co_await xendv::concurrent_for_each(
      first, last, [&](const basit::index_range& rng) noexcept -> xena::future<> {
        co_await fold_generators_partial(g_vector_p, g_vector, decomposition, rng);
      });
}

//--------------------------------------------------------------------------------------------------
// fold_generators
//--------------------------------------------------------------------------------------------------
xena::future<void> fold_generators(basct::span<c21t::element_p3> g_vector_p,
                                   basct::cspan<c21t::element_p3> g_vector,
                                   basct::cspan<unsigned> decomposition) noexcept {

  // Pick some reasonable values for min and max chunk size so that
  // we don't run out of GPU memory or split computations that are
  // too small.
  //
  // Note: These haven't been informed by much benchmarking. I'm
  // sure there are better values. This is just putting in some
  // ballpark estimates to get started.
  size_t min_chunk_size = 1ull << 9u;
  size_t max_chunk_size = 1ull << 18u;
  return fold_generators_impl(g_vector_p, g_vector, decomposition, basdv::get_num_devices(),
                              min_chunk_size, max_chunk_size);
}
} // namespace sxt::prfip
