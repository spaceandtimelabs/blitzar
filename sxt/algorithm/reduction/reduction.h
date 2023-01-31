#pragma once

#include <cassert>
#include <concepts>

#include "sxt/algorithm/base/identity_mapper.h"
#include "sxt/algorithm/base/mapper.h"
#include "sxt/algorithm/base/reducer.h"
#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/algorithm/reduction/thread_reduction.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/execution/async/computation_handle.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/base/stream.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/execution/kernel/launch.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"

namespace sxt::algr {
//--------------------------------------------------------------------------------------------------
// host_reduce
//--------------------------------------------------------------------------------------------------
namespace detail {
template <algb::reducer Reducer, algb::mapper Reader>
  requires std::same_as<typename Reducer::value_type, typename Reader::value_type>
xena::future<typename Reducer::value_type> host_reduce(Reader mapper, unsigned int n) noexcept {
  using T = typename Reducer::value_type;
  xena::computation_handle handle;
  xenb::stream stream;
  memmg::managed_array<char> buffer{n * Reader::num_bytes_per_index, memr::get_pinned_resource()};
  auto host_mapper = mapper.async_make_host_mapper(buffer.data(), stream, n, 0);
  handle.add_stream(std::move(stream));
  auto on_completion = [buffer = std::move(buffer), host_mapper = host_mapper,
                        n = n](T& value) noexcept {
    host_mapper.map_index(value, 0);
    for (unsigned int i = 1; i < n; ++i) {
      Reducer::accumulate(value, host_mapper.map_index(i));
    }
  };
  return xena::future<T>{std::move(handle), std::move(on_completion)};
}
} // namespace detail

//--------------------------------------------------------------------------------------------------
// reduction_kernel
//--------------------------------------------------------------------------------------------------
template <algb::reducer Reducer, unsigned int BlockSize, algb::mapper Reader>
  requires std::same_as<typename Reducer::value_type, typename Reader::value_type>
__global__ void reduction_kernel(typename Reducer::value_type* out, Reader mapper, unsigned int n) {
  using T = typename Reducer::value_type;
  auto thread_index = threadIdx.x;
  auto block_index = blockIdx.x;
  auto index = block_index * (BlockSize * 2) + thread_index;
  auto step = BlockSize * 2 * gridDim.x;
  __shared__ T shared_data[BlockSize];
  thread_reduce<Reducer, BlockSize>(out + block_index, shared_data, mapper, n, step, thread_index,
                                    index);
}

//--------------------------------------------------------------------------------------------------
// reduce
//--------------------------------------------------------------------------------------------------
template <algb::reducer Reducer, class Reader>
xena::future<typename Reducer::value_type> reduce(Reader mapper, unsigned int n) noexcept {
  using T = typename Reducer::value_type;
  auto dims = fit_reduction_kernel(n);
  if (dims.num_blocks == 0) {
    return detail::host_reduce<Reducer>(mapper, n);
  }

  // kernel computation
  xena::computation_handle handle;
  xenb::stream stream;
  memmg::managed_array<T> out_array{dims.num_blocks, memr::get_device_resource()};
  xenk::launch_kernel(
      dims.block_size,
      [&]<unsigned int BlockSize>(std::integral_constant<unsigned int, BlockSize>) noexcept {
        reduction_kernel<Reducer, BlockSize>
            <<<dims.num_blocks, BlockSize, 0, stream.raw_stream()>>>(out_array.data(), mapper, n);
      });
  memmg::managed_array<T> result_array{dims.num_blocks, memr::get_pinned_resource()};
  basdv::async_memcpy_device_to_host(result_array.data(), out_array.data(),
                                     sizeof(T) * dims.num_blocks, stream);
  handle.add_stream(std::move(stream));

  // completion
  auto on_completion = [
                           // clang-format off
    out_array = std::move(out_array), 
    result_array = std::move(result_array),
    num_blocks = dims.num_blocks
                           // clang-format on
  ](T& value) noexcept {
    value = result_array[0];
    for (unsigned int i = 1; i < num_blocks; ++i) {
      Reducer::accumulate(value, result_array[i]);
    }
  };

  return xena::future<T>{std::move(handle), std::move(on_completion)};
}
} // namespace sxt::algr
