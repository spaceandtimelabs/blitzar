#pragma once

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/type/raw_stream.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// reduce_output 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void reduce_output(T* __restrict__ reduction, const T* __restrict__ products,
                                 unsigned n) noexcept {
  T res = products[n - 1];
  --n;
  while (n-- > 0) {
    double_element(res, res);
    auto e = products[n];
    add_inplace(res, e);
  }
  *reduction = res;
}

//--------------------------------------------------------------------------------------------------
// reduce_products
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void reduce_products(basct::span<T> reductions, bast::raw_stream_t stream, basct::cspan<T> products) noexcept {
  auto num_outputs = reductions.size();
  auto reduction_size = products.size() / reductions.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_active_device_pointer(reductions.data()) &&
      products.size() == reduction_size * num_outputs &&
      basdv::is_active_device_pointer(products.data())
      // clang-format on
  );
  auto f = [
               // clang-format off
    reductions = reductions.data(),
    products = products.data(),
    reduction_size = reduction_size
               // clang-format on
  ] __device__
           __host__(unsigned /*num_outputs*/, unsigned output_index) noexcept {
             reduce_output(reductions + output_index, products + output_index * reduction_size,
                           reduction_size);
           };
  algi::launch_for_each_kernel(stream, f, num_outputs);
}
} // namespace sxt::mtxpp2
