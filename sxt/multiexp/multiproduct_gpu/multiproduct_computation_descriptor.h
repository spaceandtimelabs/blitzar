#pragma once

#include "sxt/execution/kernel/block_size.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/multiproduct_gpu/block_computation_descriptor.h"

namespace sxt::mtxmpg {
//--------------------------------------------------------------------------------------------------
// multiproduct_computation_descriptor
//--------------------------------------------------------------------------------------------------
struct multiproduct_computation_descriptor {
  unsigned num_blocks;
  xenk::block_size_t max_block_size;
  memmg::managed_array<block_computation_descriptor> block_descriptors;

  auto operator<=>(const multiproduct_computation_descriptor&) const noexcept = default;
};
} // namespace sxt::mtxmpg
