#include "sxt/multiexp/pippenger/multiproduct_decomposition_gpu.h"

#include <algorithm>
#include <memory>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/synchronization.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/pippenger/multiproduct_decomposition_kernel.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// rearrange_indexes
//--------------------------------------------------------------------------------------------------
static xena::future<> rearrange_indexes(basct::span<unsigned> indexes, bast::raw_stream_t stream,
                                        unsigned element_num_bits, basct::cspan<unsigned> indexes_p,
                                        basct::cspan<unsigned> block_counts) noexcept {
  auto n = indexes_p.size() / element_num_bits;
  auto num_blocks = block_counts.size() / element_num_bits;
  auto k = basn::divide_up(n, num_blocks);
  auto out = indexes.data();
  for (unsigned bit_index = 0; bit_index < element_num_bits; ++bit_index) {
    for (unsigned block_index = 0; block_index < num_blocks; ++block_index) {
      auto cnt = block_counts[block_index * element_num_bits + bit_index];
      if (cnt == 0) {
        continue;
      }
      auto bit_step = std::min(n - k * block_index, k);
      auto src = indexes_p.data() + k * block_index * element_num_bits + bit_index * bit_step;
      basdv::async_copy_device_to_device(basct::span<unsigned>{out, cnt},
                                         basct::cspan<unsigned>{src, cnt}, stream);
      out += cnt;
    }
  }
  return xena::await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct_decomposition
//--------------------------------------------------------------------------------------------------
xena::future<>
compute_multiproduct_decomposition(memmg::managed_array<unsigned>& indexes,
                                   basct::span<unsigned> product_sizes, bast::raw_stream_t stream,
                                   const mtxb::exponent_sequence& exponents) noexcept {
  auto element_num_bytes = exponents.element_nbytes;
  auto element_num_bits = 8u * element_num_bytes;
  SXT_DEBUG_ASSERT(
      // clang-format off
      product_sizes.size() == element_num_bits &&
      basdv::is_host_pointer(product_sizes.data())
      // clang-format on
  );
  auto n = exponents.n;
  std::fill(product_sizes.begin(), product_sizes.end(), 0);
  indexes.reset();
  if (n == 0) {
    co_return;
  }

  memr::async_device_resource resource{stream};

  memmg::managed_array<unsigned> indexes_p{n * element_num_bits, &resource};
  memmg::managed_array<unsigned> block_counts;
  co_await decompose_exponent_bits(indexes_p, block_counts, stream, exponents);

  // compute product sizes
  SXT_DEBUG_ASSERT(block_counts.size() % element_num_bits == 0);
  unsigned num_blocks = block_counts.size() / element_num_bits;
  unsigned num_one_bits = 0;
  for (unsigned block_index = 0; block_index < num_blocks; ++block_index) {
    for (unsigned bit_index = 0; bit_index < element_num_bits; ++bit_index) {
      auto cnt = block_counts[block_index * element_num_bits + bit_index];
      product_sizes[bit_index] += cnt;
      num_one_bits += cnt;
    }
  }
  if (num_one_bits == 0) {
    co_return;
  }

  // rearrange indexes
  indexes = memmg::managed_array<unsigned>{
      num_one_bits,
      indexes.get_allocator(),
  };
  SXT_DEBUG_ASSERT(basdv::is_device_pointer(indexes.data()));
  co_await rearrange_indexes(indexes, stream, element_num_bits, indexes_p, block_counts);
}
} // namespace sxt::mtxpi
