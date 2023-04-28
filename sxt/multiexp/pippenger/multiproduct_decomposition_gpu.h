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
// compute_multiproduct_decomposition
//--------------------------------------------------------------------------------------------------
xena::future<> compute_multiproduct_decomposition(memmg::managed_array<int>& indexes,
                                                  basct::span<unsigned> product_sizes,
                                                  bast::raw_stream_t stream,
                                                  mtxb::exponent_sequence exponents) noexcept;
} // namespace sxt::mtxpi
