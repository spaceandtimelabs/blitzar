/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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
#include "sxt/multiexp/bucket_method2/multiproduct_table.h"

/*
 * This is a workaround to define _VSTD before including cub/cub.cuh.
 * It should be removed when we can upgrade to a newer version of CUDA.
 *
 * We need to define _VSTD in order to use the clang version defined in 
 * clang.nix and the CUDA toolkit version defined in cuda.nix.
 * 
 * _VSTD was deprecated and removed from the LLVM truck.
 * NVIDIA: https://github.com/NVIDIA/cccl/pull/1331
 * LLVM: https://github.com/llvm/llvm-project/commit/683bc94e1637bd9bacc978f5dc3c79cfc8ff94b9
 * 
 * We cannot currently use any CUDA toolkit above 12.4.1 because the Kubernetes 
 * cluster currently cannot install a driver above 550. 
 * 
 * See CUDA toolkit and driver support: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
 */
#include <__config>
#define _VSTD std::_LIBCPP_ABI_NAMESPACE
_LIBCPP_BEGIN_NAMESPACE_STD _LIBCPP_END_NAMESPACE_STD

#include "cub/cub.cuh"
#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/log/log.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/base/scalar_array.h"
#include "sxt/multiexp/bucket_method2/multiproduct_table_kernel.h"

namespace sxt::mtxbk2 {
//--------------------------------------------------------------------------------------------------
// make_multiproduct_table
//--------------------------------------------------------------------------------------------------
xena::future<> make_multiproduct_table(basct::span<uint16_t> bucket_prefix_counts,
                                       basct::span<uint16_t> indexes,
                                       basct::cspan<const uint8_t*> scalars,
                                       unsigned element_num_bytes, unsigned bit_width,
                                       unsigned n) noexcept {
  auto num_outputs = scalars.size();
  auto num_buckets_per_digit = (1u << bit_width) - 1u;
  auto num_digits = basn::divide_up(element_num_bytes * 8u, bit_width);
  auto num_buckets_per_output = num_buckets_per_digit * num_digits;
  auto num_buckets_total = num_buckets_per_output * num_outputs;
  SXT_DEBUG_ASSERT(bucket_prefix_counts.size() == num_buckets_total &&
                   indexes.size() == num_outputs * num_digits * n &&
                   basdv::is_active_device_pointer(bucket_prefix_counts.data()) &&
                   basdv::is_active_device_pointer(indexes.data()));

  // transpose scalars
  basl::info("copying scalars to device");
  memmg::managed_array<uint8_t> bytes{num_outputs * n * element_num_bytes,
                                      memr::get_device_resource()};
  co_await mtxb::transpose_scalars_to_device(bytes, scalars, element_num_bytes, n);

  // compute buckets
  basl::info("computing multiproduct decomposition");
  SXT_RELEASE_ASSERT(bit_width == 8u, "only support bit_width == 8u for now");
  SXT_RELEASE_ASSERT(n <= max_multiexponentiation_length_v, "limit length for now");
  basdv::stream stream;
  fit_multiproduct_table_kernel(
      [&]<unsigned NumThreads, unsigned ItemsPerThread>(
          std::integral_constant<unsigned, NumThreads>,
          std::integral_constant<unsigned, ItemsPerThread>) noexcept {
        multiproduct_table_kernel<NumThreads, ItemsPerThread, 8>
            <<<dim3(num_digits, num_outputs, 1), NumThreads, 0, stream>>>(
                bucket_prefix_counts.data(), indexes.data(), bytes.data(), n);
      },
      n);

  // prefix sum
  auto f = [bucket_prefix_counts = bucket_prefix_counts.data(),
            num_buckets_per_digit = num_buckets_per_digit] __host__
           __device__(unsigned /*num_digits_total*/, unsigned index) noexcept {
             auto counts = bucket_prefix_counts + index * num_buckets_per_digit;
             for (unsigned i = 1; i < num_buckets_per_digit; ++i) {
               counts[i] += counts[i - 1u];
             }
           };
  algi::launch_for_each_kernel(stream, f, num_digits * num_outputs);
  co_await xendv::await_stream(stream);
}
} // namespace sxt::mtxbk2
