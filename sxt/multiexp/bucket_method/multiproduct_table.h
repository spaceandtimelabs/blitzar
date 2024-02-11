#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// make_multiproduct_table 
//--------------------------------------------------------------------------------------------------
xena::future<> make_multiproduct_table(basct::span<unsigned> bucket_prefix_counts,
                                       memmg::managed_array<unsigned>& indexes,
                                       basct::cspan<const uint8_t*> scalars,
                                       unsigned element_num_bytes, unsigned bit_width,
                                       unsigned n) noexcept;
} // namespace sxt::mtxb
