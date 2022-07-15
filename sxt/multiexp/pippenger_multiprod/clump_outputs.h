#pragma once

#include <cstdint>
#include <vector>

#include "sxt/base/container/span.h"

namespace sxt::mtxi {
class index_table;
}

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// compute_clumped_output_table
//--------------------------------------------------------------------------------------------------
bool compute_clumped_output_table(mtxi::index_table& table, std::vector<uint64_t>& output_clumps,
                                  basct::cspan<basct::cspan<uint64_t>> rows,
                                  size_t num_active_inputs, size_t clump_size) noexcept;

inline bool compute_clumped_output_table(mtxi::index_table& table,
                                         std::vector<uint64_t>& output_clumps,
                                         basct::span<basct::span<uint64_t>> rows,
                                         size_t num_active_inputs, size_t clump_size) noexcept {
  return compute_clumped_output_table(table, output_clumps,
                                      reinterpret_cast<basct::cspan<basct::cspan<uint64_t>>&>(rows),
                                      num_active_inputs, clump_size);
}

//--------------------------------------------------------------------------------------------------
// rewrite_multiproducts_with_output_clumps
//--------------------------------------------------------------------------------------------------
void rewrite_multiproducts_with_output_clumps(basct::span<basct::span<uint64_t>> rows,
                                              basct::cspan<uint64_t> output_clumps,
                                              size_t clump_size) noexcept;
} // namespace sxt::mtxpmp
