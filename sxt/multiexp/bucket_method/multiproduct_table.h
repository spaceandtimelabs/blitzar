#pragma once

#include "sxt/base/container/span.h"
#include "sxt/memory/management/managed_array_fwd.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::mtxbk {
struct bucket_descriptor;

//--------------------------------------------------------------------------------------------------
// compute_multiproduct_table
//--------------------------------------------------------------------------------------------------
xena::future<void> compute_multiproduct_table(memmg::managed_array<bucket_descriptor>& table,
                                              memmg::managed_array<unsigned>& indexes,
                                              basct::cspan<uint8_t> scalars,
                                              unsigned element_num_bytes,
                                              unsigned bit_width) noexcept;
} // namespace sxt::mtxbk
