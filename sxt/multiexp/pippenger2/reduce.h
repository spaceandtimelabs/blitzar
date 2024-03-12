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
// reduce_product 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void reduce_product(T* __restrict__ reduction, const T* __restrict__ products,
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
// reduce_product 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void reduce_product(basct::span<T> reductions, bast::raw_stream_t stream, basct::cspan<T> products,
                    unsigned element_num_bytes) noexcept {
  auto num_outputs = reductions.size();
  auto reduction_size = element_num_bytes * 8u;
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_active_device_pointer(reductions.data()) &&
      products.size() == reduction_size * num_outputs &&
      basdv::is_active_device_pointer(products.data())
      // clang-format on
  );
  (void)reductions;
  (void)products;
}
} // namespace sxt::mtxpp2
