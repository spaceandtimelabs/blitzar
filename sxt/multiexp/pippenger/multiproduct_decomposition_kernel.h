#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/type/raw_stream.h"
#include "sxt/execution/async/future_fwd.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// decompose_exponent_bits
//--------------------------------------------------------------------------------------------------
xena::future<> decompose_exponent_bits(basct::span<unsigned> indexes_p,
                                       memmg::managed_array<unsigned>& block_counts,
                                       bast::raw_stream_t stream,
                                       const mtxb::exponent_sequence& exponents) noexcept;
} // namespace sxt::mtxpi
