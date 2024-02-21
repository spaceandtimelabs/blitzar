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
#pragma once

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// reduce_digits_rest
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void reduce_digits_rest(T& reduction, T& t, const T* __restrict__ sums,
                                      unsigned num_buckets_per_digit) noexcept {
  unsigned i = num_buckets_per_digit - 1u;
  while (i-- > 0) {
    auto e = sums[i];
    add_inplace(t, e);
    add(reduction, reduction, t);
  }
}

//--------------------------------------------------------------------------------------------------
// reduction_kernel
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void reduction_kernel(T* __restrict__ res, const T* __restrict__ sums,
                                    unsigned num_digits, unsigned bit_width,
                                    unsigned output_index) noexcept {
  auto num_buckets_per_digit = (1u << bit_width) - 1u;

  // adjust pointers
  res += output_index;
  sums += output_index * num_digits * num_buckets_per_digit;

  // reduce
  unsigned digit_index = num_digits - 1u;
  T t = sums[num_digits * num_buckets_per_digit - 1u];
  T reduction = t;
  reduce_digits_rest(reduction, t, sums + digit_index * num_buckets_per_digit,
                     num_buckets_per_digit);
  while (digit_index-- > 0) {
    for (unsigned i = 0; i < bit_width; ++i) {
      double_element(reduction, reduction);
    }
    t = sums[(digit_index + 1u) * num_buckets_per_digit - 1u];
    add(reduction, reduction, t);
    reduce_digits_rest(reduction, t, sums + digit_index * num_buckets_per_digit,
                       num_buckets_per_digit);
  }

  // write result
  *res = reduction;
}

//--------------------------------------------------------------------------------------------------
// reduce_buckets
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> reduce_buckets(basct::span<T> res, basct::cspan<T> bucket_sums,
                              unsigned element_num_bytes, unsigned bit_width) noexcept {
  auto num_outputs = res.size();
  auto num_buckets_per_digit = (1u << bit_width) - 1u;
  auto num_digits = basn::divide_up(element_num_bytes * 8u, bit_width);
  SXT_DEBUG_ASSERT(
      // clang-format off
      res.size() == num_outputs &&
      basdv::is_host_pointer(res.data()) &&
      bucket_sums.size() == num_outputs * num_digits * num_buckets_per_digit &&
      basdv::is_active_device_pointer(bucket_sums.data())
      // clang-format on
  );
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> res_dev{num_outputs, &resource};
  auto f = [
               // clang-format off
    res = res_dev.data(),
    sums = bucket_sums.data(),
    num_digits = num_digits,
    bit_width = bit_width
               // clang-format on
  ] __host__
           __device__(unsigned /*num_outputs*/, unsigned output_index) noexcept {
             reduction_kernel(res, sums, num_digits, bit_width, output_index);
           };
  algi::launch_for_each_kernel(stream, f, num_outputs);
  basdv::async_copy_device_to_host(res, res_dev, stream);
  co_await xendv::await_stream(stream);
}
} // namespace sxt::mtxbk
