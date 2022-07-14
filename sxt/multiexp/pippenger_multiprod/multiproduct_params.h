#pragma once

#include <cstddef>

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// multiproduct_params
//--------------------------------------------------------------------------------------------------
struct multiproduct_params {
  size_t partition_size;
  size_t input_clump_size;
  size_t output_clump_size;
};
} // namespace sxt::mtxpmp
