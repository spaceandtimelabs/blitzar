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
// decomposition_kernel
//--------------------------------------------------------------------------------------------------
static __global__ void decomposition_kernel(unsigned* out, unsigned* counters, const uint8_t* data,
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
  auto bit_step = umin(k, last - first);

  data += first * element_num_bytes + byte_index;
  out += block_index * element_num_bits * k + bit_index * bit_step;

  const auto mask = 1u << bit_offset;
  unsigned count = 0;
  for (unsigned i = first; i < last; ++i) {
    out[count] = i;
    auto byte = *data;
    count += static_cast<unsigned>((byte & mask) != 0);
    data += element_num_bytes;
  }
  counters[block_index * element_num_bits + bit_index] = count;
}

//--------------------------------------------------------------------------------------------------
// decompose_exponent_bits
//--------------------------------------------------------------------------------------------------
xena::future<> decompose_exponent_bits(basct::span<unsigned> indexes_p,
                                       memmg::managed_array<unsigned>& block_counts,
                                       bast::raw_stream_t stream,
                                       const mtxb::exponent_sequence& exponents) noexcept {
  unsigned element_num_bytes = exponents.element_nbytes;
  unsigned element_num_bits = 8u * element_num_bytes;
  auto n = exponents.n;
  SXT_DEBUG_ASSERT(
      // clang-format off
      indexes_p.size() == n * element_num_bits &&
      basdv::is_device_pointer(indexes_p.data())
      // clang-format on
  );
  auto num_blocks = std::min(n, 128ul);

  block_counts = memmg::managed_array<unsigned>{
      num_blocks * element_num_bits,
      block_counts.get_allocator(),
  };
  SXT_DEBUG_ASSERT(basdv::is_host_pointer(block_counts.data()));

  memr::async_device_resource resource{stream};

  // set up exponents
  memmg::managed_array<uint8_t> exponents_data{&resource};
  auto data = exponents.data;
  if (!basdv::is_device_pointer(exponents.data)) {
    exponents_data = memmg::managed_array<uint8_t>{
        n * element_num_bytes,
        &resource,
    };
    basdv::async_memcpy_host_to_device(exponents_data.data(), data, n * element_num_bytes, stream);
    data = exponents_data.data();
  }

  // set up block_counts_dev
  memmg::managed_array<unsigned> block_counts_dev{block_counts.size(), &resource};

  // launch kernel
  decomposition_kernel<<<dim3(num_blocks, 1, 1), dim3(element_num_bytes, 8, 1), 0, stream>>>(
      indexes_p.data(), block_counts_dev.data(), data, static_cast<unsigned>(n));
  exponents_data.reset();

  // transfer counts back to host
  basdv::async_copy_device_to_host(block_counts, block_counts_dev, stream);

  // set up future
  return xena::await_stream(stream);
}
} // namespace sxt::mtxpi
