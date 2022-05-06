#pragma once

#include "sxt/memory/management/managed_array.h"

namespace sxt::sqcb { class commitment; }

namespace sxt {
//--------------------------------------------------------------------------------------------------
// multi_commitment_cpu
//--------------------------------------------------------------------------------------------------
void multi_commitment_cpu(
    memmg::managed_array<sqcb::commitment> &commitments_per_col,
    uint64_t rows, uint64_t cols, uint64_t element_nbytes,
    const memmg::managed_array<uint8_t> &data_table) noexcept;

} // namespace sxt
