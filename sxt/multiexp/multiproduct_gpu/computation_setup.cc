#include "sxt/multiexp/multiproduct_gpu/computation_setup.h"

#include <algorithm>
#include <limits>

#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/multiproduct_gpu/multiproduct_computation_descriptor.h"

namespace sxt::mtxmpg {
//--------------------------------------------------------------------------------------------------
// fill_block_descriptors
//--------------------------------------------------------------------------------------------------
static void fill_block_descriptors(basct::span<block_computation_descriptor> descriptors,
                                   basct::cspan<unsigned> product_sizes,
                                   basct::cspan<xenk::kernel_dims> reduction_dims) noexcept {
  unsigned index_count = 0;
  unsigned block_count = 0;
  auto out_iter = descriptors.begin();
  for (size_t product_index = 0; product_index < product_sizes.size(); ++product_index) {
    auto n = product_sizes[product_index];
    auto dims = reduction_dims[product_index];
    block_computation_descriptor descriptor{
        .block_offset = block_count,
        .index_first = index_count,
        .n = n,
        .reduction_num_blocks = dims.num_blocks,
        .block_size = dims.block_size,
    };
    out_iter = std::fill_n(out_iter, dims.num_blocks, descriptor);
    index_count += n;
    block_count += dims.num_blocks;
  }
}

//--------------------------------------------------------------------------------------------------
// setup_multiproduct_computation
//--------------------------------------------------------------------------------------------------
void setup_multiproduct_computation(multiproduct_computation_descriptor& descriptor,
                                    basct::cspan<unsigned> product_sizes) noexcept {
  size_t entry_count = 0;
  size_t block_count = 0;
  memmg::managed_array<xenk::kernel_dims> reduction_dims(product_sizes.size());
  descriptor.max_block_size = xenk::block_size_t::v1;
  for (size_t product_index = 0; product_index < product_sizes.size(); ++product_index) {
    auto n = product_sizes[product_index];
    SXT_DEBUG_ASSERT(n > 0);
    entry_count += n;
    auto dims = algr::fit_reduction_kernel(n);
    block_count += dims.num_blocks;
    descriptor.max_block_size = std::max(descriptor.max_block_size, dims.block_size);
    reduction_dims[product_index] = dims;
  }
  SXT_RELEASE_ASSERT(entry_count <= std::numeric_limits<unsigned>::max(),
                     "we don't support reductions this large");

  descriptor.num_blocks = block_count;

  // block_descriptors
  memmg::managed_array<block_computation_descriptor> block_descriptors(
      block_count, descriptor.block_descriptors.get_allocator());
  fill_block_descriptors(block_descriptors, product_sizes, reduction_dims);
  descriptor.block_descriptors = std::move(block_descriptors);
}
} // namespace sxt::mtxmpg
