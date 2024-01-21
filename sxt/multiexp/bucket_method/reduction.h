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
  auto sum = bucket_sums[--bucket_last];
  auto partial = sum;
  while (bucket_last != bucket_first) {
    auto e = bucket_sums[--bucket_last];
    add_inplace(sum, e);
    add(partial, partial, sum);
  }
  reductions[2u * reduction_index] = sum;
  reductions[2u * reduction_index + 1u] = partial;
}

//--------------------------------------------------------------------------------------------------
// complete_bucket_group_reduction_kernel 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void complete_bucket_group_reduction_kernel(T* __restrict__ reductions,
                                                          const T* __restrict__ partial_reductions,
                                                          unsigned reduction_width_log2,
                                                          unsigned num_partials_per_group,
                                                          unsigned group_index) noexcept {
  auto partial_first = num_partials_per_group * group_index;
  auto partial_last = partial_first + num_partials_per_group;
  --partial_last;
  T sum_part1 = partial_reductions[2u * partial_last];
  T t = sum_part1; 
  T sum_part2 = partial_reductions[2u * partial_last + 1u];
  while (partial_last != partial_first) {
    --partial_last;
    auto e = partial_reductions[2u * partial_last];
    add_inplace(t, e);
    add(sum_part1, sum_part1, t);
    e = partial_reductions[2u * partial_last + 1u];
    add_inplace(sum_part2, e);
  }
  for (unsigned i=0; i<reduction_width_log2; ++i) {
    double_element(sum_part1, sum_part1);
  }
  add_inplace(sum_part1, sum_part2);
  reductions[group_index] = sum_part1;
}

//--------------------------------------------------------------------------------------------------
// partial_reduce
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void partial_reduce(basct::span<T> reductions, const basdv::stream& stream,
                    basct::cspan<T> bucket_sums, unsigned bit_width,
                    unsigned reduction_width) noexcept {
  auto num_buckets_per_group = (1u << bit_width) - 1u;
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
// complete_bucket_group_reduction 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void complete_bucket_group_reduction(basct::span<T> reductions, basct::cspan<T> partial_reductions,
                                     const basdv::stream& stream,
                                     unsigned reduction_width_log2) noexcept {
  auto num_groups = reductions.size();
  auto num_partials_per_group = partial_reductions.size() / num_groups / 2u;
  auto f =
      [
          // clang-format off
    reductions = reductions.data(),
    partial_reductions = partial_reductions.data(),
    reduction_width_log2 = reduction_width_log2,
    num_partials_per_group = num_partials_per_group
          // clang-format on
  ] __device__ __host__(unsigned /*num_groups*/, unsigned group_index) {
        complete_bucket_group_reduction_kernel(reductions, partial_reductions, reduction_width_log2,
                                               num_partials_per_group, group_index);
      };
  algi::launch_for_each_kernel(stream, f, num_groups);
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
  (void)reductions;
  (void)bucket_sums;
  (void)bit_width;
  (void)reduction_width;
#if 0
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
#endif
}
} // namespace sxt::mtxbk
