#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/execution/async/future.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// multiexponentiate 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> multiexponentiate(basct::span<T> res, const partition_table_accessor<T>& accessor,
                                 basct::cspan<uint8_t> scalars) noexcept {
  (void)res;
  (void)accessor;
  (void)scalars;
  return {};
}
} // namespace sxt::mtxpp2
