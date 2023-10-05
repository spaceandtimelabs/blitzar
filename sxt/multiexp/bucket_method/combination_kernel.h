#pragma once

#include "sxt/base/curve/element.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// combine_partial_bucket_sums
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
__global__ void combine_partial_bucket_sums(T* out, T* partial_bucket_sums, unsigned num_partials) {
  auto bucket_group_size = gridDim.x;
  auto bucket_group_index = threadIdx.x;
  auto num_bucket_groups = blockDim.x;
  auto bucket_index = blockIdx.x;
  auto output_index = blockIdx.y;

  partial_bucket_sums += bucket_index + bucket_group_size * bucket_group_index +
                         bucket_group_size * num_bucket_groups * num_partials * output_index;

  out += bucket_index + bucket_group_size * bucket_group_index +
         bucket_group_size * num_bucket_groups * output_index;

  T sum = *partial_bucket_sums;
  partial_bucket_sums += bucket_group_size * num_bucket_groups;
  for (unsigned i = 1; i < num_partials; ++i) {
    add_inplace(sum, *partial_bucket_sums);
    partial_bucket_sums += bucket_group_size * num_bucket_groups;
  }

  *out = sum;
}

//--------------------------------------------------------------------------------------------------
// combine_bucket_groups
//--------------------------------------------------------------------------------------------------
template <unsigned BucketGroupSize, unsigned NumBucketGroups, bascrv::element T>
__global__ void combine_bucket_groups(T* out, T* bucket_sums) {
  auto thread_index = threadIdx.x;
  auto block_index = blockIdx.x;
  auto num_threads = blockDim.x;
  auto bucket_index = thread_index + block_index * num_threads;
  if (bucket_index >= BucketGroupSize) {
    return;
  }

  auto output_index = blockIdx.y;

  bucket_sums += bucket_index;
  out += bucket_index + BucketGroupSize * NumBucketGroups * output_index;

  unsigned i = NumBucketGroups - 1;
  T sum = bucket_sums[i * BucketGroupSize];
  while (i-- > 0) {
    double_element(sum, sum);
    add_inplace(sum, bucket_sums[BucketGroupSize * i]);
  }

  *out = sum;
}
} // namespace sxt::mtxbk
