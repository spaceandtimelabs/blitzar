#pragma once

#include <numeric>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/property.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/memory/management/managed_array.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// combine_reduce 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
template <bascrv::element T>
xena::future<> combine_reduce(basct::span<T> res, basct::cspan<unsigned> output_bit_table,
                              basct::cspan<T> partial_products) noexcept {
  auto num_outputs = output_bit_table.size();

  // partials
  memmg::managed_array<unsigned> bit_table_partial_sums(num_outputs);
  std::partial_sum(output_bit_table.begin(), output_bit_table.end(),
                   bit_table_partial_sums.begin());
  auto reduction_size = partial_products.size() / bit_table_partial_sums[num_outputs - 1];

  // split
  auto [chunk_first, chunk_last] =
      basit::split(basit::index_range{0, num_outputs}, basdv::get_num_devices());

  // combine reduce

}
#pragma clang diagnostic pop
} // namespace sxt::mtxpp2
