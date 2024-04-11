#pragma once

#include <memory>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// make_in_memory_partition_table_accessor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
std::unique_ptr<partition_table_accessor<T>>
make_in_memory_partition_table_accessor(basct::cspan<T> generators) noexcept {
  auto n = generators.size();
  std::vector<T> generators_data;
  if (n % 16 != 0) {
    n = basn::divide_up(n, size_t{16}) * 16u;
    generators_data.resize(n);
    auto iter = std::copy(generators.begin(), generators.end(), generators_data.begin());
    std::fill(iter, generators_data.end(), T::identity());
    generators = generators_data;
  }
  (void)generators;
  return {};
}
} // namespace sxt:mtxpp2
