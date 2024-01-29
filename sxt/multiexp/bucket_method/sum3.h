#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/device/device_viewable.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/digit_utility.h"
#include "sxt/multiexp/base/scalar_array.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// bucket_sum_kernel 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
__global__ void bucket_sum_kernel(T* __restrict__ partial_sums, const T* __restrict__ generators,
                                  const uint8_t* __restrict__ scalars, unsigned element_num_bytes,
                                  unsigned bit_width, unsigned n) noexcept {
  (void)partial_sums;
  (void)generators;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
  (void)n;
}

//--------------------------------------------------------------------------------------------------
// compute_bucket_sums
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void compute_bucket_sums(basct::span<T> sums, const basdv::stream& stream,
                         basct::cspan<T> generators, basct::cspan<const uint8_t*> scalars,
                         unsigned element_num_bytes, unsigned bit_width) noexcept {
  (void)sums;
  (void)stream;
  (void)generators;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
}
} // namespace sxt::mtxbk
