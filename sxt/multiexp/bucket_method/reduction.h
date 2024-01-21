#pragma once

#include <algorithm>
#include <cassert>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// reduce_bucket_group 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void reduce_bucket_group(T* __restrict__ reductions,
                                       const T* __restrict__ bucket_sums, unsigned bit_width,
                                       unsigned num_buckets_per_output, unsigned reduction_width,
                                       unsigned reduction_index) noexcept {
  auto num_reductions_per_output = basn::divide_up(num_buckets_per_output, reduction_width);
  auto output_index = reduction_index / num_reductions_per_output;
  auto bucket_first = reduction_index % num_reductions_per_output;
  auto bucket_last = std::min(bucket_first + reduction_width, num_buckets_per_output);
  bucket_first += output_index * num_buckets_per_output;
  bucket_last += output_index * num_buckets_per_output;
  assert(bucket_last - bucket_first > 1 && "there must be at least two buckets to reduce");
  T res = bucket_sums[--bucket_last];
  while (bucket_first != bucket_last) {
    for (unsigned i = 0; i < bit_width; ++i) {
      double_element(res, res);
    }
    auto e = bucket_sums[--bucket_last];
    add_inplace(res, e);
  }
  reductions[reduction_index] = res;
}

template <bascrv::element T>
CUDA_CALLABLE void reduce_bucket_group(T* __restrict__ reductions,
                                       const T* __restrict__ bucket_sums, unsigned bit_width,
                                       unsigned reduction_width,
                                       unsigned reduction_index) noexcept {
  auto num_buckets_per_group = (1u << bit_width) - 1u;
  auto num_reductions_per_group = basn::divide_up(num_buckets_per_group, reduction_width);
  auto group_index = reduction_index / num_reductions_per_group;
  auto reduction_group_index = reduction_index % num_reductions_per_group;
  auto bucket_first_group = group_index * num_buckets_per_group;
  auto bucket_first = bucket_first_group + reduction_group_index * reduction_width;
  auto bucket_last =
      std::min(bucket_first + reduction_width, bucket_first_group + num_buckets_per_group);
  auto res = bucket_sums[--bucket_last];
  while (bucket_last != bucket_first) {
    add_inplace(res, res, bucket_sums[--bucket_last]);
  }
  reductions[reduction_index] = res;
}

//--------------------------------------------------------------------------------------------------
// compute_partial_reduction_device
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void compute_partial_reduction_device(basct::span<T> reductions, const basdv::stream& stream,
                                      basct::cspan<T> bucket_sums, unsigned bit_width,
                                      unsigned num_outputs, unsigned reduction_width) noexcept {
  auto num_buckets = bucket_sums.size();
  auto num_buckets_per_output = num_buckets / num_outputs;
  auto num_reductions_per_output = basn::divide_up(num_buckets_per_output, reduction_width);
  auto num_reductions = num_reductions_per_output * num_outputs;
  if (reduction_width == 1) {
    basdv::async_copy_device_to_host(reductions, bucket_sums, stream);
    return;
  }
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> reductions_dev{num_reductions, &resource};
  auto f = [
               // clang-format off
    reductions = reductions_dev.data(),
    bucket_sums = bucket_sums,
    bit_width = bit_width,
    num_buckets_per_output = num_buckets_per_output,
    reduction_width = reduction_width
               // clang-format on
  ] __device__
           __host__(unsigned /*num_reductions*/, unsigned reduction_index) noexcept {
             reduce_bucket_group(reductions, bucket_sums, bit_width, num_buckets_per_output,
                                 reduction_width, reduction_index);
           };
  algi::launch_for_each_kernel(stream, f, num_reductions);
  basdv::async_copy_device_to_host(reductions, reductions_dev, stream);
}

template <bascrv::element T>
void compute_partial_reduction_device(basct::span<T> reductions, const basdv::stream& stream,
                                      basct::cspan<T> bucket_sums, unsigned bit_width,
                                      unsigned reduction_width) noexcept {
  auto num_buckets_per_group = (1u << bit_width) - 1u;
  auto num_reductions_per_group = basn::divide_up(num_buckets_per_group, reduction_width);
  auto num_bucket_groups = bucket_sums.size() / num_buckets_per_group;
  auto num_reductions = num_bucket_groups * num_buckets_per_group;
  auto f =
      [
          // clang-format off
    reductions = reductions.data(),
    bucket_sums = bucket_sums.data(),
    bit_width = bit_width,
    reduction_width = reduction_width
          // clang-format on
  ] __device__
      __host__(unsigned /*num_reductions*/, unsigned reduction_index) noexcept {
        reduce_bucket_group(reductions, bucket_sums, bit_width, reduction_width, reduction_index);
      };
  algi::launch_for_each_kernel(stream, f, num_reductions);
}

//--------------------------------------------------------------------------------------------------
// compute_partial_reduction_host
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void compute_partial_reduction_host(basct::span<T> reductions, basct::cspan<T> bucket_sums,
                                    unsigned bit_width, unsigned num_outputs,
                                    unsigned reduction_width) noexcept {
  auto num_buckets = bucket_sums.size();
  auto num_buckets_per_output = num_buckets / num_outputs;
  auto num_reductions_per_output = basn::divide_up(num_buckets_per_output, reduction_width);
  auto num_reductions = num_reductions_per_output * num_outputs;
  for (unsigned reduction_index = 0; reduction_index < num_reductions; ++reduction_index) {
    reduce_bucket_group(reductions.data(), bucket_sums.data(), bit_width, num_buckets_per_output,
                        reduction_width, reduction_index);
  }
}

//--------------------------------------------------------------------------------------------------
// plan_reduction 
//--------------------------------------------------------------------------------------------------
unsigned plan_reduction(unsigned num_buckets, unsigned num_outputs) noexcept;

//--------------------------------------------------------------------------------------------------
// compute_reduction
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> compute_reduction(basct::span<T> reductions, basct::cspan<T> bucket_sums,
                                 unsigned bit_width, unsigned reduction_width = 0) noexcept {
  auto num_buckets = bucket_sums.size();
  auto num_outputs = reductions.size();
  auto num_buckets_per_output = num_buckets / num_outputs;
  if (reduction_width == 0) {
    reduction_width = plan_reduction(num_buckets, num_outputs);
  }
  basdv::stream stream;
  if (reduction_width == num_buckets_per_output) {
    compute_partial_reduction_device(reductions, stream, bucket_sums, bit_width, num_outputs,
                                     reduction_width);
    co_return co_await xendv::await_stream(std::move(stream));
  }
  auto num_reductions_p = basn::divide_up(num_buckets_per_output, reduction_width) * num_outputs;
  memmg::managed_array<T> partial_reductions{num_reductions_p};
  compute_partial_reduction_device(partial_reductions, stream, bucket_sums, bit_width, num_outputs,
                                   reduction_width);
  co_await xendv::await_stream(std::move(stream));
  compute_partial_reduction_host(reductions, partial_reductions, bit_width, num_outputs,
                                 num_reductions_p / num_outputs);
}
} // namespace sxt::mtxbk
