#include "sxt/proof/sumcheck/mle_utility.h"
 
#include <algorithm>
#include <iostream>

#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// copy_partial_mles 
//--------------------------------------------------------------------------------------------------
void copy_partial_mles(memmg::managed_array<s25t::element>& partial_mles, basdv::stream& stream,
                       basct::cspan<s25t::element> mles, unsigned n, unsigned a,
                       unsigned b) noexcept {
  auto num_variables = std::max(basn::ceil_log2(n), 1);
  auto mid = 1u << (num_variables - 1u);
  auto num_mles = mles.size() / n;
  auto part1_size = b - a;
  /* SXT_DEBUG_ASSERT( */
  /*     b <= mid */
  /* ); */
  auto ap = std::min(mid + a, n);
  auto bp = std::min(mid + b, n);
  auto part2_size = bp - ap;
   
  // resize array
  auto partial_length = part1_size + part2_size;
  std::cerr << "n = " << n << std::endl;
  std::cerr << "num_mles = " << num_mles << std::endl;
  std::cerr << "part1 = " << part1_size << std::endl;
  std::cerr << "part2 = " << part2_size << std::endl;
  partial_mles.resize(partial_length * num_mles);
   
  // copy data
  for (unsigned mle_index=0; mle_index<num_mles; ++mle_index) {
    // first part
    auto src = mles.subspan(n * mle_index + a, part1_size);
    auto dst = basct::subspan(partial_mles, partial_length * mle_index, part1_size);
    basdv::async_copy_host_to_device(dst, src, stream);

    // second part
    src = mles.subspan(n * mle_index + ap, part2_size);
    dst = basct::subspan(partial_mles, partial_length * mle_index + part1_size, part2_size);
    if (!src.empty()) {
      basdv::async_copy_host_to_device(dst, src, stream);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// copy_folded_mles 
//--------------------------------------------------------------------------------------------------
void copy_folded_mles(basct::span<s25t::element> host_mles, basdv::stream& stream,
                      basct::cspan<s25t::element> device_mles, unsigned np, unsigned a,
                      unsigned b) noexcept {
  auto num_mles = host_mles.size() / np;
  auto slice_n = device_mles.size() / num_mles;
  auto slice_np = b - a;
  std::cout << "copy_folded_mles: np = " << np << std::endl;
  std::cout << "copy_folded_mles: a, b = " << a << " " << b << std::endl;
  std::cout << "copy_folded_mles: num_mles = " << num_mles << std::endl;
  std::cout << "copy_folded_mles: slice_n = " << slice_n << std::endl;
  std::cout << "copy_folded_mles: slice_np = " << slice_np << std::endl;
  std::cout << "copy_folded_mles: host_mles.size() = " << host_mles.size() << std::endl;
  std::cout << "copy_folded_mles: device_mles.size() = " << device_mles.size() << std::endl;
  SXT_DEBUG_ASSERT(
      host_mles.size() == num_mles * np &&
      device_mles.size() == num_mles * slice_n &&
      b <= np
  );
  for (unsigned mle_index=0; mle_index<num_mles; ++mle_index) {
    auto src = device_mles.subspan(mle_index * slice_n, slice_np);
    auto dst = host_mles.subspan(mle_index * np + a, slice_np);
    basdv::async_copy_device_to_host(dst, src, stream);
  }
}
} // namespace sxt::prfsk