#include "sxt/multiexp/multiproduct_gpu/computation_setup.h"

#include <algorithm>
#include <limits>

#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/multiproduct_gpu/multiproduct_computation_descriptor.h"

namespace sxt::mtxmpg {
//--------------------------------------------------------------------------------------------------
// fill_block_descriptors
//--------------------------------------------------------------------------------------------------
static void fill_block_descriptors(basct::span<block_computation_descriptor> descriptors,
                                   basct::cspan<basct::cspan<uint64_t>> products,
                                   basct::cspan<xenk::kernel_dims> reduction_dims) noexcept {
  unsigned index_count = 0;
  unsigned block_count = 0;
  auto out_iter = descriptors.begin();
  for (size_t reduction_index = 0; reduction_index < products.size(); ++reduction_index) {
    auto n = static_cast<unsigned>(products[reduction_index].size() - 2);
    auto dims = reduction_dims[reduction_index];
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
// fill_indexes_impl
//--------------------------------------------------------------------------------------------------
template <class F>
static void fill_indexes_impl(basct::span<unsigned> indexes,
                              basct::cspan<basct::cspan<uint64_t>> products, F remapper) noexcept {
  auto out_iter = indexes.begin();
  for (auto product : products) {
    for (auto index : product.subspan(2)) {
      *out_iter++ = remapper(index);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// fill_indexes
//--------------------------------------------------------------------------------------------------
static void fill_indexes(basct::span<unsigned> indexes,
                         basct::cspan<basct::cspan<uint64_t>> products,
                         const basct::blob_array& masks, size_t num_inputs) noexcept {
  if (num_inputs == masks.size()) {
    return fill_indexes_impl(indexes, products,
                             [](uint64_t index) noexcept { return static_cast<unsigned>(index); });
  }
  size_t count = 0;
  memmg::managed_array<unsigned> remap_array(num_inputs);
  for (unsigned index_p = 0; index_p < masks.size(); ++index_p) {
    auto mask = masks[index_p];
    auto is_active =
        std::any_of(mask.begin(), mask.end(), [](uint8_t b) noexcept { return b != 0; });
    if (is_active) {
      remap_array[count++] = static_cast<unsigned>(index_p);
    }
  }
  fill_indexes_impl(indexes, products, [&](uint64_t index) noexcept { return remap_array[index]; });
}

//--------------------------------------------------------------------------------------------------
// setup_multiproduct_computation
//--------------------------------------------------------------------------------------------------
void setup_multiproduct_computation(multiproduct_computation_descriptor& descriptor,
                                    basct::cspan<basct::cspan<uint64_t>> products,
                                    const basct::blob_array& masks, size_t num_inputs) noexcept {
  SXT_DEBUG_ASSERT(num_inputs <= masks.size());
  SXT_RELEASE_ASSERT(masks.size() <= std::numeric_limits<unsigned>::max(),
                     "we don't support reductions this large");
  size_t entry_count = 0;
  size_t block_count = 0;
  memmg::managed_array<xenk::kernel_dims> reduction_dims(products.size());
  descriptor.max_block_size = xenk::block_size_t::v1;
  for (size_t reduction_index = 0; reduction_index < products.size(); ++reduction_index) {
    SXT_DEBUG_ASSERT(products[reduction_index].size() > 2);
    auto n = products[reduction_index].size() - 2;
    entry_count += n;
    auto dims = algr::fit_reduction_kernel(static_cast<unsigned>(n));
    block_count += dims.num_blocks;
    descriptor.max_block_size = std::max(descriptor.max_block_size, dims.block_size);
    reduction_dims[reduction_index] = dims;
  }
  SXT_RELEASE_ASSERT(entry_count <= std::numeric_limits<unsigned>::max(),
                     "we don't support reductions this large");

  descriptor.num_blocks = block_count;

  // block_descriptors
  memmg::managed_array<block_computation_descriptor> block_descriptors(
      block_count, descriptor.block_descriptors.get_allocator());
  fill_block_descriptors(block_descriptors, products, reduction_dims);
  descriptor.block_descriptors = std::move(block_descriptors);

  // indexes
  memmg::managed_array<unsigned> indexes(entry_count, descriptor.indexes.get_allocator());
  fill_indexes(indexes, products, masks, num_inputs);
  descriptor.indexes = std::move(indexes);
}
} // namespace sxt::mtxmpg
