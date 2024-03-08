#pragma once

#include "sxt/base/curve/element.h"
#include "sxt/base/container/span.h"
#include "sxt/execution/async/future.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// multiexponentiate 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> multiexponentiate(basct::span<T> res, basct::cspan<T> partition_table,
                                 basct::cspan<const uint8_t*> exponents,
                                 unsigned element_num_bytes) noexcept {
  // make the index table
  // compute sums from indexes
  // reduce
  (void)res;
  (void)partition_table;
  (void)exponents;
  (void)element_num_bytes;
  return {};
}
} // namespace sxt::mtxpp2
