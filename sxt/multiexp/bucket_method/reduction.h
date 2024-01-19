#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/stream.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// compute_partial_reduction 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void compute_partial_reduction(basct::span<T> reductions, const basdv::stream& stream,
                               basct::cspan<T> bucket_sums, unsigned bit_width,
                               unsigned num_buckets, unsigned num_outputs,
                               unsigned reduction_width) noexcept {
  (void)reductions;
  (void)stream;
  (void)bucket_sums;
  (void)bit_width;
  (void)num_buckets;
  (void)num_outputs;
  (void)reduction_width;
}
} // namespace sxt::mtxbk
