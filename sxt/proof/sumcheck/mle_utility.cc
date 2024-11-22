#include "sxt/proof/sumcheck/mle_utility.h"
 
#include <algorithm>

#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// copy_partial_mles 
//--------------------------------------------------------------------------------------------------
void copy_partial_mles(memmg::managed_array<s25t::element>& partial_mles, basdv::stream& stream,
                       basct::cspan<s25t::element> mles, unsigned n, unsigned a,
                       unsigned b) noexcept {
  auto mid = std::max(n / 2u, 1u);
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
void copy_folded_mles(basct::span<s25t::element> host_mles, basdv::stream& stream,
                      basct::cspan<s25t::element> device_mles, unsigned n, unsigned a,
                      unsigned b) noexcept {
  auto num_mles = host_mles.size() / n;
  auto slice_size = device_mles.size() / num_mles;
  auto np = n / 2u;
  SXT_DEBUG_ASSERT(
      host_mles.size() == num_mles * n &&
      device_mles.size() == num_mles * slice_size &&
      b <= np
  );
#if 0
  for (index_t mle_index=0; mle_index<num_mles; ++mle_index) {
    /* auto src = device_mles.subspan(mle_index */
  }
#endif
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
