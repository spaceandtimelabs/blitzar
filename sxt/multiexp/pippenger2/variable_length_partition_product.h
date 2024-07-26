#pragma once

#include <concepts>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// async_partition_product
//--------------------------------------------------------------------------------------------------
/**
 * Compute the multiproduct for the bits of an array of scalars using an accessor to
 * precomputed sums for each group of generators.
 */
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<> async_partition_product(basct::span<T> products,
                                       const partition_table_accessor<U>& accessor,
                                       basct::cspan<uint8_t> scalars, 
                                       basct::cspan<unsigned> lengths,
                                       unsigned offset) noexcept {
  (void)products;
  (void)accessor;
  (void)scalars;
  (void)lengths;
  (void)offset;
  return {};
}
} // namespace sxt::mtxpp2
