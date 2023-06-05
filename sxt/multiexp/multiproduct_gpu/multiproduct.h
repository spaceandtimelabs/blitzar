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
#pragma once

#include <concepts>
#include <memory>

#include "sxt/algorithm/base/mapper.h"
#include "sxt/algorithm/base/reducer.h"
#include "sxt/base/container/span.h"
#include "sxt/base/device/event.h"
#include "sxt/base/device/event_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/multiproduct_gpu/completion.h"
#include "sxt/multiexp/multiproduct_gpu/computation_setup.h"
#include "sxt/multiexp/multiproduct_gpu/kernel.h"
#include "sxt/multiexp/multiproduct_gpu/multiproduct_computation_descriptor.h"

namespace sxt::mtxmpg {
//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
template <algb::reducer Reducer, algb::mapper Mapper>
  requires std::same_as<typename Reducer::value_type, typename Mapper::value_type> &&
           std::constructible_from<Mapper, const typename Reducer::value_type*, const unsigned*>
xena::future<> compute_multiproduct(basct::span<typename Reducer::value_type> products,
                                    bast::raw_stream_t stream,
                                    basct::cspan<typename Reducer::value_type> generators,
                                    basct::cspan<unsigned> indexes,
                                    basct::cspan<unsigned> product_sizes) noexcept {
  auto num_products = products.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      products.size() == num_products &&
      product_sizes.size() == num_products &&
      basdv::is_host_pointer(products.data()) &&
      basdv::is_device_pointer(generators.data()) &&
      basdv::is_device_pointer(indexes.data()) &&
      basdv::is_host_pointer(product_sizes.data())
      // clang-format on
  );
  using T = typename Reducer::value_type;
  multiproduct_computation_descriptor computation_descriptor{
      .num_blocks{},
      .max_block_size{},
      .block_descriptors{memr::get_pinned_resource()},
  };
  setup_multiproduct_computation(computation_descriptor, product_sizes);

  memr::async_device_resource resource{stream};

  // block_descriptors_gpu
  memmg::managed_array<block_computation_descriptor> block_descriptors_gpu{
      computation_descriptor.block_descriptors.size(), &resource};
  basdv::async_copy_host_to_device(block_descriptors_gpu, computation_descriptor.block_descriptors,
                                   stream);

  // launch kernel
  memmg::managed_array<T> partial_res_gpu{computation_descriptor.num_blocks, &resource};
  auto max_block_size = static_cast<unsigned>(computation_descriptor.max_block_size);
  auto shared_memory = sizeof(T) * max_block_size * 2;
  multiproduct_kernel<Reducer, Mapper>
      <<<computation_descriptor.num_blocks, max_block_size, shared_memory, stream>>>(
          partial_res_gpu.data(), generators.data(), indexes.data(), block_descriptors_gpu.data());

  // partial_res
  memmg::managed_array<T> partial_res{partial_res_gpu.size(), memr::get_pinned_resource()};
  basdv::async_copy_device_to_host(partial_res, partial_res_gpu, stream);

  // future
  // clang-format off
  return xendv::await_stream(stream).then([
      products,
      block_descriptors = std::move(computation_descriptor.block_descriptors),
      partial_res = std::move(partial_res)
  ]() noexcept { 
    complete_multiproduct<Reducer>(products, block_descriptors, partial_res); 
  });
  // clang-format on
}
} // namespace sxt::mtxmpg
