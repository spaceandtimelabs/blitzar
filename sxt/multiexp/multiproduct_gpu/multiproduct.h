#pragma once

#include "sxt/algorithm/base/reducer.h"
#include "sxt/base/container/span.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/execution/async/computation_handle.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/synchronization.h"
#include "sxt/execution/base/stream.h"
#include "sxt/memory/management/managed_array.h"
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
template <algb::reducer Reducer>
xena::future<memmg::managed_array<typename Reducer::value_type>>
compute_multiproduct(basct::cspan<typename Reducer::value_type> generators,
                     memmg::managed_array<unsigned>&& indexes,
                     basct::cspan<unsigned> product_sizes) noexcept {
  using T = typename Reducer::value_type;
  multiproduct_computation_descriptor computation_descriptor{
      .num_blocks{},
      .max_block_size{},
      .indexes{memr::get_pinned_resource()},
      .block_descriptors{memr::get_pinned_resource()},
  };
  setup_multiproduct_computation(computation_descriptor, std::move(indexes), product_sizes);
  xenb::stream stream;

  // indexes_gpu
  memmg::managed_array<unsigned> indexes_gpu{computation_descriptor.indexes.size(),
                                             memr::get_device_resource()};
  basdv::async_copy_host_to_device(indexes_gpu, computation_descriptor.indexes, stream);

  // block_descriptors_gpu
  memmg::managed_array<block_computation_descriptor> block_descriptors_gpu{
      computation_descriptor.block_descriptors.size(), memr::get_device_resource()};
  basdv::async_copy_host_to_device(block_descriptors_gpu, computation_descriptor.block_descriptors,
                                   stream);

  // launch kernel
  memmg::managed_array<T> partial_res_gpu{computation_descriptor.num_blocks,
                                          memr::get_device_resource()};
  auto max_block_size = static_cast<unsigned>(computation_descriptor.max_block_size);
  auto shared_memory = sizeof(T) * max_block_size;
  multiproduct_kernel<Reducer>
      <<<computation_descriptor.num_blocks, max_block_size, shared_memory, stream>>>(
          partial_res_gpu.data(), generators.data(), indexes_gpu.data(),
          block_descriptors_gpu.data());

  // partial_res
  memmg::managed_array<T> partial_res{partial_res_gpu.size(), memr::get_pinned_resource()};
  basdv::async_copy_device_to_host(partial_res, partial_res_gpu, stream);

  // completion
  auto completion = [
                        // clang-format off
    computation_descriptor = std::move(computation_descriptor),
    indexes_gpu = std::move(indexes_gpu),
    block_descriptors_gpu = std::move(block_descriptors_gpu),
    partial_res_gpu = std::move(partial_res_gpu),
    num_products = product_sizes.size()
                        // clang-format on
  ](memmg::managed_array<T>&& partial_res) noexcept {
    memmg::managed_array<T> res(num_products);
    complete_multiproduct<Reducer>(res, computation_descriptor.block_descriptors, partial_res);
    return res;
  };
  return xena::await_and_own_stream(std::move(stream), std::move(partial_res)).then(completion);
}
} // namespace sxt::mtxmpg
