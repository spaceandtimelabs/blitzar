#pragma once

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/execution/async/coroutine.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// reduce_digit
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void reduce_digit_rest(T& reduction, T& t, const T* __restrict__ sums,
                                     unsigned num_buckets_per_digit) noexcept {
  unsigned i = num_buckets_per_digit - 1u;
  while (i-- > 0) {
    add_inplace(t, sums[i]);
    add(reduction, reduction, t);
  }
}

//--------------------------------------------------------------------------------------------------
// reduction_buckets 
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
  reduce_rest(reduction, t, sums + digit_index * num_buckets_per_digit, num_buckets_per_digit);
  while (digit_index-- > 0) {
    for (unsigned i=0; i<bit_width; ++i) {
      double_element(reduction, reduction);
    }
    t = sums[(digit_index + 1u) * num_buckets_per_digit - 1u];
    add(reduction, reduction, t);
    reduce_rest(reduction, t, sums + digit_index * num_buckets_per_digit, num_buckets_per_digit);
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
  (void)res;
  (void)bucket_sums;
  (void)element_num_bytes;
  (void)bit_width;
  co_return;
}
} // namespace sxt::mtxbk
