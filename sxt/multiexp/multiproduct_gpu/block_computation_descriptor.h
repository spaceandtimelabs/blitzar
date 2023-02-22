#pragma once

#include <cstdint>

#include "sxt/execution/kernel/block_size.h"

namespace sxt::mtxmpg {
//--------------------------------------------------------------------------------------------------
// block_computation_descriptor
//--------------------------------------------------------------------------------------------------
struct block_computation_descriptor {
  unsigned block_offset;
  unsigned index_first;
  unsigned n;
  unsigned reduction_num_blocks;
  xenk::block_size_t block_size;

  auto operator<=>(const block_computation_descriptor&) const noexcept = default;
};
} // namespace sxt::mtxmpg
