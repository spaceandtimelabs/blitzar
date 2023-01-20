#include "sxt/multiexp/pippenger_multiprod/test_driver.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "sxt/base/bit/iteration.h"
#include "sxt/base/container/span_void.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/error/panic.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/clump2_descriptor.h"
#include "sxt/multiexp/index/clump2_marker_utility.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// apply_partition_operation
//--------------------------------------------------------------------------------------------------
void test_driver::apply_partition_operation(basct::span_void inout,
                                            basct::cspan<uint64_t> partition_markers,
                                            size_t partition_size) const noexcept {
  auto num_inputs = inout.size();
  auto num_inputs_p = partition_markers.size();

  basct::span<uint64_t> inputs{static_cast<uint64_t*>(inout.data()), num_inputs};
  memmg::managed_array<uint64_t> inputs_p(num_inputs_p);

  for (size_t marker_index = 0; marker_index < num_inputs_p; ++marker_index) {
    auto marker = partition_markers[marker_index];
    auto partition_index = marker >> partition_size;
    auto bitset = marker ^ (partition_index << partition_size);
    SXT_DEBUG_ASSERT(bitset != 0);
    size_t partition_first = partition_index * partition_size;
    uint64_t reduction = 0;
    while (bitset != 0) {
      auto index = basbt::consume_next_bit(bitset);
      reduction += inputs[partition_first + index];
    }
    inputs_p[marker_index] = reduction;
  }

  std::copy_n(inputs_p.data(), inputs_p.size(), inputs.data());
}

//--------------------------------------------------------------------------------------------------
// apply_clump2_operation
//--------------------------------------------------------------------------------------------------
void test_driver::apply_clump2_operation(basct::span_void inout, basct::cspan<uint64_t> markers,
                                         const mtxi::clump2_descriptor& descriptor) const noexcept {
  auto num_inputs = inout.size();
  auto num_inputs_p = markers.size();

  basct::span<uint64_t> inputs{static_cast<uint64_t*>(inout.data()), num_inputs};
  memmg::managed_array<uint64_t> inputs_p(num_inputs_p);

  for (size_t marker_index = 0; marker_index < num_inputs_p; ++marker_index) {
    auto marker = markers[marker_index];
    uint64_t clump_index, index1, index2;
    mtxi::unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    auto clump_first = descriptor.size * clump_index;
    auto reduction = inputs[clump_first + index1];
    SXT_DEBUG_ASSERT(index1 <= index2);
    if (index1 != index2) {
      reduction += inputs[clump_first + index2];
    }
    inputs_p[marker_index] = reduction;
  }

  std::copy_n(inputs_p.data(), num_inputs_p, inputs.data());
}

//--------------------------------------------------------------------------------------------------
// compute_naive_multiproduct
//--------------------------------------------------------------------------------------------------
void test_driver::compute_naive_multiproduct(basct::span_void inout,
                                             basct::cspan<basct::cspan<uint64_t>> products,
                                             size_t num_inactive_inputs) const noexcept {
  basct::span<uint64_t> inputs{static_cast<uint64_t*>(inout.data()), inout.size()};
  memmg::managed_array<uint64_t> outputs(products.size());
  for (auto row : products) {
    SXT_DEBUG_ASSERT(row.size() >= 2 && row.size() >= 2 + row[1]);
    auto& output = outputs[row[0]];
    output = 0;
    auto num_inactive_entries = row[1];
    // add inactive entries
    for (size_t index = 2; index < 2 + num_inactive_entries; ++index) {
      output += inputs[row[index]];
    }
    // add active entries
    for (size_t index = 2 + num_inactive_entries; index < row.size(); ++index) {
      output += inputs[num_inactive_inputs + row[index]];
    }
  }
  std::copy_n(outputs.data(), outputs.size(), inputs.data());
}

//--------------------------------------------------------------------------------------------------
// permute_inputs
//--------------------------------------------------------------------------------------------------
void test_driver::permute_inputs(basct::span_void inout,
                                 basct::cspan<uint64_t> permutation) const noexcept {
  auto n = permutation.size();
  if (n > inout.size()) {
    baser::panic("n > inout.size()");
  }
  basct::span<uint64_t> inputs{static_cast<uint64_t*>(inout.data()), n};
  memmg::managed_array<uint64_t> inputs_p(n);
  for (size_t index = 0; index < n; ++index) {
    auto index_p = permutation[index];
    if (index_p >= n) {
      baser::panic("index_p >= n");
    }
    inputs_p[index] = inputs[index_p];
  }
  std::copy_n(inputs_p.data(), n, inputs.data());
}
} // namespace sxt::mtxpmp
