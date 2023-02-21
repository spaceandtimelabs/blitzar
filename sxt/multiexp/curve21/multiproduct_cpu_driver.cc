#include "sxt/multiexp/curve21/multiproduct_cpu_driver.h"

#include <algorithm>
#include <vector>

#include "sxt/base/container/span_void.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/bitset_multiprod/multiproduct.h"
#include "sxt/multiexp/bitset_multiprod/value_cache.h"
#include "sxt/multiexp/bitset_multiprod/value_cache_utility.h"
#include "sxt/multiexp/curve21/multiproduct_bitset_operator.h"
#include "sxt/multiexp/index/clump2_descriptor.h"
#include "sxt/multiexp/index/clump2_marker_utility.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// apply_partition_operation
//--------------------------------------------------------------------------------------------------
void multiproduct_cpu_driver::apply_partition_operation(basct::span_void inout,
                                                        basct::cspan<uint64_t> partition_markers,
                                                        size_t partition_size) const noexcept {
  auto num_inputs = inout.size();
  auto num_inputs_p = partition_markers.size();

  basct::span<c21t::element_p3> inputs{static_cast<c21t::element_p3*>(inout.data()), num_inputs};
  memmg::managed_array<c21t::element_p3> inputs_p(num_inputs_p);

  std::vector<c21t::element_p3> cache_data(mtxbmp::compute_cache_size(partition_size));
  mtxbmp::value_cache<c21t::element_p3> cache;
  multiproduct_bitset_operator op;
  uint64_t partition_index = static_cast<uint64_t>(-1);
  for (size_t marker_index = 0; marker_index < num_inputs_p; ++marker_index) {
    auto marker = partition_markers[marker_index];
    auto partition_index_p = marker >> partition_size;
    size_t partition_first = partition_index_p * partition_size;
    if (partition_index_p != partition_index) {
      partition_index = partition_index_p;
      cache = {cache_data.data(), std::min(partition_size, num_inputs - partition_first)};
      mtxbmp::init_value_cache(
          cache, op,
          basct::cspan<c21t::element_p3>{inputs.subspan(partition_first, cache.num_terms())});
    }
    auto bitset = marker ^ (partition_index << partition_size);
    mtxbmp::compute_multiproduct(inputs_p[marker_index], cache, op, bitset);
  }
  std::copy_n(inputs_p.data(), inputs_p.size(), inputs.data());
}

//--------------------------------------------------------------------------------------------------
// apply_clump2_operation
//--------------------------------------------------------------------------------------------------
void multiproduct_cpu_driver::apply_clump2_operation(
    basct::span_void inout, basct::cspan<uint64_t> markers,
    const mtxi::clump2_descriptor& descriptor) const noexcept {
  auto num_inputs = inout.size();
  auto num_inputs_p = markers.size();

  basct::span<c21t::element_p3> inputs{static_cast<c21t::element_p3*>(inout.data()), num_inputs};
  memmg::managed_array<c21t::element_p3> inputs_p(num_inputs_p);

  for (size_t marker_index = 0; marker_index < num_inputs_p; ++marker_index) {
    auto marker = markers[marker_index];
    uint64_t clump_index, index1, index2;
    mtxi::unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    auto clump_first = descriptor.size * clump_index;
    auto reduction = inputs[clump_first + index1];

    SXT_DEBUG_ASSERT(index1 <= index2);

    if (index1 != index2) {
      c21o::add(reduction, reduction, inputs[clump_first + index2]);
    }

    inputs_p[marker_index] = reduction;
  }

  std::copy_n(inputs_p.data(), num_inputs_p, inputs.data());
}

//--------------------------------------------------------------------------------------------------
// compute_naive_multiproduct
//--------------------------------------------------------------------------------------------------
void multiproduct_cpu_driver::compute_naive_multiproduct(
    basct::span_void inout, basct::cspan<basct::cspan<uint64_t>> products,
    size_t num_inactive_inputs) const noexcept {
  basct::span<c21t::element_p3> inputs{static_cast<c21t::element_p3*>(inout.data()), inout.size()};
  memmg::managed_array<c21t::element_p3> outputs(products.size());

  for (auto row : products) {
    SXT_DEBUG_ASSERT(row.size() >= 2 && row.size() >= 2 + row[1]);
    auto& output = outputs[row[0]];

    if (row.size() == 2) {
      output = c21cn::zero_p3_v;
      continue;
    }

    size_t active_index;
    auto num_inactive_entries = row[1];

    if (num_inactive_entries > 0) {
      output = inputs[row[2]];
      active_index = num_inactive_entries + 2;

      // add inactive entries
      for (size_t inactive_index = 3; inactive_index < 2 + num_inactive_entries; ++inactive_index) {
        c21o::add(output, output, inputs[row[inactive_index]]);
      }
    } else {
      active_index = 3;
      output = inputs[num_inactive_inputs + row[2]];
    }

    // add active entries
    for (; active_index < row.size(); ++active_index) {
      c21o::add(output, output, inputs[num_inactive_inputs + row[active_index]]);
    }
  }

  std::copy_n(outputs.data(), outputs.size(), inputs.data());
}

//--------------------------------------------------------------------------------------------------
// permute_inputs
//--------------------------------------------------------------------------------------------------
void multiproduct_cpu_driver::permute_inputs(basct::span_void inout,
                                             basct::cspan<uint64_t> permutation) const noexcept {
  auto n = permutation.size();
  SXT_DEBUG_ASSERT(n <= inout.size());

  memmg::managed_array<c21t::element_p3> inputs_p(n);
  basct::span<c21t::element_p3> inputs{static_cast<c21t::element_p3*>(inout.data()), n};

  for (size_t index = 0; index < n; ++index) {
    auto index_p = permutation[index];
    inputs_p[index] = inputs[index_p];
  }

  std::copy_n(inputs_p.data(), n, inputs.data());
}
} // namespace sxt::mtxc21
