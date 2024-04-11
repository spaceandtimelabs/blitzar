#pragma once

#include <memory>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// make_in_memory_partition_table_accessor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
std::unique_ptr<partition_table_accessor<T>>
make_in_memory_partition_table_accessor(basct::cspan<T> generators) noexcept {
  (void)generators;
  return {};
}
} // namespace sxt:mtxpp2
