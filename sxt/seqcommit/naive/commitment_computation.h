#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxb { struct exponent_sequence; }
namespace sxt::sqcb { class commitment; }

namespace sxt::sqcnv {

void fill_data(uint8_t a_i[32], const uint8_t *bytes_row_i_column_k, uint8_t size_row_data) noexcept;

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void compute_commitments(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<mtxb::exponent_sequence> value_sequences) noexcept;
}  // namespace sxt::sqcnv
