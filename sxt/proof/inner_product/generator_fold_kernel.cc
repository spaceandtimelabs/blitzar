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
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/curve21/type/element_p3.h"
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
  auto n = g_vector.size();
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

  // f

  (void)g_vector;
  (void)g_vector_p;
  (void)decomposition;
  return {};
}

//--------------------------------------------------------------------------------------------------
// fold_generators
//--------------------------------------------------------------------------------------------------
xena::future<void> fold_generators(basct::span<c21t::element_p3> g_vector,
                                   basct::cspan<unsigned> decomposition) noexcept {
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_active_device_pointer(g_vector.data()) &&
      basdv::is_host_pointer(decomposition.data()) &&
      g_vector.size() % 2 == 0 && 
      g_vector.size() > 1
      // clang-format on
  );
  auto n = static_cast<unsigned>(g_vector.size() / 2);
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<unsigned> decomposition_gpu{decomposition.size(), &resource};
  basdv::async_copy_host_to_device(decomposition_gpu, decomposition, stream);
  auto data = g_vector.data();
  auto decomposition_data = decomposition_gpu.data();
  auto decomposition_size = static_cast<unsigned>(decomposition.size());
  auto f = [
               // clang-format off
    data,
    decomposition_data,
    decomposition_size
               // clang-format on
  ] __device__
           __host__(unsigned n, unsigned i) noexcept {
             fold_generators(data[i],
                             basct::cspan<unsigned>{decomposition_data, decomposition_size},
                             data[i], data[i + n]);
           };
  return algi::for_each(std::move(stream), f, n);
}

xena::future<void> fold_generators(basct::span<c21t::element_p3> g_vector_p,
                                   basct::cspan<c21t::element_p3> g_vector,
                                   basct::cspan<unsigned> decomposition) noexcept {
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
  (void)g_vector_p;
  (void)g_vector;
  (void)decomposition;
  (void)fold_generators_partial;
  return {};
}
} // namespace sxt::prfip
