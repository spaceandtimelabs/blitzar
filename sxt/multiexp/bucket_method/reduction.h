#pragma once

#include <algorithm>
#include <cassert>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// partially_reduce_bucket_group_kernel
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void
partially_reduce_bucket_group_kernel(T* __restrict__ reductions, const T* __restrict__ bucket_sums,
                                     unsigned bit_width, unsigned reduction_width,
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
  while (partial_last != partial_first + 1) {
    --partial_last;
    auto e = partial_reductions[2u * partial_last];
    add_inplace(t, e);
    add(sum_part1, sum_part1, t);
    e = partial_reductions[2u * partial_last + 1u];
    add_inplace(sum_part2, e);
  }
  --partial_last;
  auto e = partial_reductions[2u * partial_last + 1u];
  add_inplace(sum_part2, e);
  for (unsigned i=0; i<reduction_width_log2; ++i) {
    double_element(sum_part1, sum_part1);
  }
  add_inplace(sum_part1, sum_part2);
  reductions[group_index] = sum_part1;
}

//--------------------------------------------------------------------------------------------------
// complete_bucket_reduction_kernel
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void complete_bucket_reduction_kernel(T* __restrict__ reductions,
                                                    const T* group_reductions, unsigned bit_width,
                                                    unsigned num_groups_per_output,
                                                    unsigned output_index) noexcept {
  auto group_first = num_groups_per_output * output_index;
  auto group_last = group_first + num_groups_per_output;
  auto res = group_reductions[--group_last];
  while (group_last != group_first) {
    for (unsigned i=0; i<bit_width; ++i) {
      double_element(res, res);
    }
    auto e = group_reductions[--group_last];
    add_inplace(res, e);
  }
  reductions[output_index] = res;
}

//--------------------------------------------------------------------------------------------------
// partially_reduce_bucket_groups
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void partially_reduce_bucket_groups(basct::span<T> reductions, const basdv::stream& stream,
                                  basct::cspan<T> bucket_sums, unsigned bit_width,
                                  unsigned reduction_width) noexcept {
  auto num_reductions = static_cast<unsigned>(reductions.size()) / 2u;
  auto f = [
               // clang-format off
    reductions = reductions.data(),
    bucket_sums = bucket_sums.data(),
    bit_width = bit_width,
    reduction_width = reduction_width
               // clang-format on
  ] __device__
           __host__(unsigned /*num_reductions*/, unsigned reduction_index) noexcept {
             partially_reduce_bucket_group_kernel(reductions, bucket_sums, bit_width,
                                                  reduction_width, reduction_index);
           };
  algi::launch_for_each_kernel(stream, f, num_reductions);
}

//--------------------------------------------------------------------------------------------------
// complete_bucket_group_reductions 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void complete_bucket_group_reductions(basct::span<T> reductions, const basdv::stream& stream,
                                      basct::cspan<T> partial_reductions,
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
  ] __device__
      __host__(unsigned /*num_groups*/, unsigned group_index) noexcept {
        complete_bucket_group_reduction_kernel(reductions, partial_reductions, reduction_width_log2,
                                               num_partials_per_group, group_index);
      };
  algi::launch_for_each_kernel(stream, f, num_groups);
}

//--------------------------------------------------------------------------------------------------
// complete_bucket_reductions
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void complete_bucket_reductions(basct::span<T> reductions, const basdv::stream& stream,
                                basct::cspan<T> group_reductions, unsigned bit_width) noexcept {
  auto num_outputs = reductions.size();
  auto num_groups_per_output = group_reductions.size() / num_outputs;
  auto f = [
               // clang-format off
    reductions = reductions.data(),
    group_reductions = group_reductions.data(),
    bit_width = bit_width,
    num_groups_per_output = num_groups_per_output
               // clang-format on
  ] __device__
           __host__(unsigned /*num_outputs*/, unsigned output_index) noexcept {
             complete_bucket_reduction_kernel(reductions, group_reductions, bit_width,
                                              num_groups_per_output, output_index);
           };
  algi::launch_for_each_kernel(stream, f, num_outputs);
}

//--------------------------------------------------------------------------------------------------
// plan_reduction 
//--------------------------------------------------------------------------------------------------
unsigned plan_reduction(unsigned bit_width, unsigned num_buckets, unsigned num_outputs) noexcept;

//--------------------------------------------------------------------------------------------------
// reduce_buckets
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void reduce_buckets(basct::span<T> reductions, const basdv::stream& stream,
                    basct::cspan<T> bucket_sums, unsigned bit_width,
                    unsigned reduction_width_log2 = 0) noexcept {
  auto num_buckets = static_cast<unsigned>(bucket_sums.size());
  auto num_outputs = static_cast<unsigned>(reductions.size());
  auto num_buckets_per_group = (1u << bit_width) - 1u;
  auto num_bucket_groups = num_buckets / num_buckets_per_group;
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_host_pointer(reductions.data()) &&
      num_outputs > 0 &&
      num_buckets > 1 &&
      num_buckets % num_outputs == 0 && 
      num_buckets % num_bucket_groups == 0 && 
      num_buckets % num_buckets_per_group == 0
      // clang-format on
  );
  if (reduction_width_log2 == 0) {
    reduction_width_log2 = plan_reduction(bit_width, num_buckets, num_outputs);
  }
  auto reduction_width = (1u << reduction_width_log2);
  auto num_partials_per_group = basn::divide_up(num_buckets_per_group, reduction_width);
  auto num_partials = num_partials_per_group * num_bucket_groups;
  memr::async_device_resource resource{stream};

  // partially reduce bucket groups
  memmg::managed_array<T> partial_group_reductions{2u * num_partials, &resource};
  partially_reduce_bucket_groups<T>(partial_group_reductions, stream, bucket_sums, bit_width,
                                 reduction_width);

  // complete bucket group reductions
  memmg::managed_array<T> group_reductions{num_bucket_groups, &resource};
  complete_bucket_group_reductions<T>(group_reductions, stream, partial_group_reductions,
                                   reduction_width_log2);
  partial_group_reductions.reset();

  // complete bucket_reductions
  memmg::managed_array<T> reductions_p{num_outputs, &resource};
  complete_bucket_reductions<T>(reductions_p, stream, group_reductions, bit_width);
  group_reductions.reset();

  basdv::async_copy_device_to_host(reductions, reductions_p, stream);
}
} // namespace sxt::mtxbk