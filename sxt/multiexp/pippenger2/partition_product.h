#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/execution/async/future.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// partition_product 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> partition_product(basct::span<T> products,
                                 const partition_table_accessor<T>& accessor,
                                 basct::cspan<uint8_t> scalars, unsigned offset) noexcept {
  (void)products;
  (void)accessor;
  (void)scalars;
  (void)offset;
  return {};
}
} // namespace sxt::mtxpp2
