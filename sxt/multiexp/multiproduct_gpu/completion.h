#pragma once

#include "sxt/algorithm/base/reducer.h"
#include "sxt/base/container/span.h"
#include "sxt/base/error/assert.h"
#include "sxt/multiexp/multiproduct_gpu/block_computation_descriptor.h"

namespace sxt::mtxmpg {
//--------------------------------------------------------------------------------------------------
// complete_multiproduct
//--------------------------------------------------------------------------------------------------
template <algb::reducer Reducer>
void complete_multiproduct(basct::span<typename Reducer::value_type> products,
                           basct::cspan<block_computation_descriptor> block_descriptors,
                           basct::cspan<typename Reducer::value_type> block_results) noexcept {
  auto num_products = products.size();
  auto num_blocks = block_descriptors.size();
  // clang-format off
  SXT_DEBUG_ASSERT(
      num_products <= num_blocks &&
      block_descriptors.size() == num_blocks &&
      block_results.size() == num_blocks 
  );
  // clang-format on
  size_t block_index = 0;
  size_t product_index = 0;
  while (block_index < num_blocks) {
    auto& descriptor = block_descriptors[block_index];
    SXT_DEBUG_ASSERT(descriptor.reduction_num_blocks > 0);
    auto product = block_results[block_index];
    for (size_t i = 1; i < descriptor.reduction_num_blocks; ++i) {
      auto block_result_i = block_results[block_index + i];
      Reducer::accumulate_inplace(product, block_result_i);
    }
    block_index += descriptor.reduction_num_blocks;
    products[product_index++] = product;
  }
  SXT_DEBUG_ASSERT(block_index == num_blocks && product_index == num_products);
}
} // namespace sxt::mtxmpg
