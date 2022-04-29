#include "sxt/multiexp/pippenger_multiprod/test_driver.h"

#include <cassert>

#include "sxt/base/bit/iteration.h"
#include "sxt/memory/management/managed_array.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// apply_partition_operation
//--------------------------------------------------------------------------------------------------
void test_driver::apply_partition_operation(
    memmg::managed_array<void>& inout, basct::cspan<uint64_t> partition_markers,
    size_t partition_size) const noexcept {
  auto num_inputs = inout.size();
  auto num_inputs_p = partition_markers.size();

  basct::cspan<int> inputs{static_cast<int*>(inout.data()), num_inputs};
  memmg::managed_array<int> inputs_p(num_inputs_p, inout.get_allocator());

  for (size_t marker_index=0; marker_index<num_inputs_p; ++marker_index) {
    auto marker = partition_markers[marker_index];
    auto partition_index = marker >> partition_size;
    auto bitset = marker ^ (partition_index << partition_size);
    assert(bitset != 0);
    size_t partition_first = partition_index * partition_size;
    int reduction = 0;
    while( bitset != 0) {
      auto index = basbt::consume_next_bit(bitset);
      reduction += inputs[partition_first + index];
    }
    inputs_p[marker_index] = reduction;
  }

  inout = std::move(inputs_p);
}
} // namespace sxt::mtxpmp
