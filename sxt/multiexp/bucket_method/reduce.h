#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/execution/async/coroutine.h"

namespace sxt::mtxbk {
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
