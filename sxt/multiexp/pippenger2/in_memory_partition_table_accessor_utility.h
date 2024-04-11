#pragma once

#include <memory>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor.h"
#include "sxt/multiexp/pippenger2/partition_table.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// make_in_memory_partition_table_accessor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
std::unique_ptr<partition_table_accessor<T>>
make_in_memory_partition_table_accessor(basct::cspan<T> generators) noexcept {
  auto n = generators.size();
  std::vector<T> generators_data;
  auto num_partitions = basn::divide_up(n, size_t{16});
  if (n % 16 != 0) {
    n = num_partitions * 16u;
    generators_data.resize(n);
    auto iter = std::copy(generators.begin(), generators.end(), generators_data.begin());
    std::fill(iter, generators_data.end(), T::identity());
    generators = generators_data;
  }
  auto num_entries = 1u << 16u;
  memmg::managed_array<T> sums{num_entries * num_partitions, memr::get_pinned_resource()};
  compute_partition_table<T>(sums, generators);
  return std::make_unique<in_memory_partition_table_accessor<T>>(std::move(sums));
}
} // namespace sxt:mtxpp2
