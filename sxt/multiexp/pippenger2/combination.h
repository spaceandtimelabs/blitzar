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
// combine_impl 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void combine_impl(T* __restrict__ reduction, const T* __restrict__ elements,
                                unsigned step, unsigned reduction_size) noexcept {
  T res = elements[0];
  for (unsigned i = 1; i < reduction_size; ++i) {
    auto e = elements[step * i];
    add_inplace(res, e);
  }
  *reduction = res;
}

//--------------------------------------------------------------------------------------------------
// combine 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void combine(basct::span<T> res, bast::raw_stream_t stream, basct::cspan<T> elements) noexcept {
  auto n = static_cast<unsigned>(res.size());
  SXT_DEBUG_ASSERT(
      // clang-format off
      elements.size() >= n && 
      elements.size() % n == 0 &&
      basdv::is_active_device_pointer(res.data()) &&
      basdv::is_active_device_pointer(elements.data())
      // clang-format on
  );
  auto reduction_size = static_cast<unsigned>(elements.size() / n);
  auto f = [
               // clang-format off
    reductions = res.data(),
    elements = elements.data(),
    reduction_size = reduction_size
               // clang-format on
  ] __device__
           __host__(unsigned n, unsigned index) noexcept {
             combine_impl(reductions + index, elements + index, n, reduction_size);
           };
  algi::launch_for_each_kernel(stream, f, n);
}
} // namespace sxt::mtxpp2
