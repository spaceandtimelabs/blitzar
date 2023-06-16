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
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/proof/inner_product/generator_fold.h"

namespace sxt::prfip {
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
} // namespace sxt::prfip
