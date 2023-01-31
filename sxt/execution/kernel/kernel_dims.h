#pragma once

#include "sxt/execution/kernel/block_size.h"

namespace sxt::xenk {
//--------------------------------------------------------------------------------------------------
// kernel_dims
//--------------------------------------------------------------------------------------------------
struct kernel_dims {
  unsigned int num_blocks = 0;
  block_size_t block_size = block_size_t::v32;
};
} // namespace sxt::xenk
