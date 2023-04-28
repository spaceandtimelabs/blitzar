#include "sxt/multiexp/pippenger/multiproduct_decomposition_kernel.h"

#include <cassert>
#include <memory>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// count_kernel
//--------------------------------------------------------------------------------------------------
static __global__ void count_kernel(unsigned* counters, const uint8_t* data, unsigned n) {
  unsigned element_num_bytes = blockDim.x;
  unsigned element_num_bits = element_num_bytes * 8u;
  unsigned num_blocks = gridDim.x;
  unsigned block_index = blockIdx.x;
  unsigned byte_index = threadIdx.x;
  unsigned bit_offset = threadIdx.y;
  unsigned bit_index = 8u * byte_index + bit_offset;

  auto k = basn::divide_up(n, num_blocks);
  auto first = k * block_index;
  assert(first < n);
  auto last = umin(first + k, n);

  data += first * element_num_bytes + byte_index;

  const auto mask = 1u << bit_offset;
  unsigned count = 0;
  for (unsigned i = first; i < last; ++i) {
    auto byte = *data;
    count += static_cast<unsigned>((byte & mask) != 0);
    data += element_num_bytes;
  }
  counters[block_index * element_num_bits + bit_index] = count;
}

//--------------------------------------------------------------------------------------------------
// decomposition_kernel
//--------------------------------------------------------------------------------------------------
static __global__ void decomposition_kernel(int* out, const unsigned* offsets, const uint8_t* data,
                                            unsigned n) {
  unsigned element_num_bytes = blockDim.x;
  unsigned element_num_bits = element_num_bytes * 8u;
  unsigned num_blocks = gridDim.x;
  unsigned block_index = blockIdx.x;
  unsigned byte_index = threadIdx.x;
  unsigned bit_offset = threadIdx.y;
  unsigned bit_index = 8u * byte_index + bit_offset;

  auto k = basn::divide_up(n, num_blocks);
  auto first = k * block_index;
  assert(first < n);
  auto last = umin(first + k, n);

  data += first * element_num_bytes + byte_index;
  out = out + offsets[block_index * element_num_bits + bit_index];

  const auto mask = 1u << bit_offset;
  unsigned count = 0;
  for (unsigned i = first; i < last; ++i) {
    auto byte = *data;
    auto is_one = static_cast<unsigned>((byte & mask) != 0);
    if (is_one == 1) {
      out[count] = static_cast<int>(i);
    }
    count += is_one;
    data += element_num_bytes;
  }
}

//--------------------------------------------------------------------------------------------------
// decompose_exponent_bits
//--------------------------------------------------------------------------------------------------
xena::future<> decompose_exponent_bits(basct::span<int> indexes, bast::raw_stream_t stream,
                                       basct::cspan<unsigned> offsets,
                                       const mtxb::exponent_sequence& exponents) noexcept {
  unsigned element_num_bytes = exponents.element_nbytes;
  unsigned element_num_bits = 8u * element_num_bytes;
  auto n = exponents.n;
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_device_pointer(indexes.data()) &&
      basdv::is_host_pointer(offsets.data()) &&
      basdv::is_device_pointer(exponents.data)
      // clang-format on
  );
  auto num_blocks = offsets.size() / element_num_bits;
  memr::async_device_resource resource{stream};

  // set up offsets_dev
  memmg::managed_array<unsigned> offsets_dev{offsets.size(), &resource};
  basdv::async_copy_host_to_device(offsets_dev, offsets, stream);

  // launch kernel
  decomposition_kernel<<<dim3(num_blocks, 1, 1), dim3(element_num_bytes, 8, 1), 0, stream>>>(
      indexes.data(), offsets_dev.data(), exponents.data, static_cast<unsigned>(n));

  // set up future
  return xena::await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// count_exponent_bits
//--------------------------------------------------------------------------------------------------
xena::future<> count_exponent_bits(memmg::managed_array<unsigned>& block_counts,
                                   bast::raw_stream_t stream,
                                   const mtxb::exponent_sequence& exponents) noexcept {
  unsigned element_num_bytes = exponents.element_nbytes;
  unsigned element_num_bits = 8u * element_num_bytes;
  auto n = exponents.n;
  SXT_DEBUG_ASSERT(basdv::is_device_pointer(exponents.data));
  memr::async_device_resource resource{stream};
  auto num_blocks = std::min(n, 128ul);
  auto num_iterations = basn::divide_up(n, num_blocks);
  num_blocks = basn::divide_up(n, num_iterations);

  block_counts = memmg::managed_array<unsigned>{
      num_blocks * element_num_bits,
      block_counts.get_allocator(),
  };
  SXT_DEBUG_ASSERT(basdv::is_host_pointer(block_counts.data()));

  // set up block_counts_dev
  memmg::managed_array<unsigned> block_counts_dev{block_counts.size(), &resource};

  // launch kernel
  count_kernel<<<dim3(num_blocks, 1, 1), dim3(element_num_bytes, 8, 1), 0, stream>>>(
      block_counts_dev.data(), exponents.data, static_cast<unsigned>(n));

  // transfer counts back to host
  basdv::async_copy_device_to_host(block_counts, block_counts_dev, stream);

  // set up future
  return xena::await_stream(stream);
}
} // namespace sxt::mtxpi
