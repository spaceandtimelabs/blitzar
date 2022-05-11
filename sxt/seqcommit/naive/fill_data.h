#pragma once

#include <cinttypes>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::sqcnv {

//--------------------------------------------------------------------------------------------------
// fill_data
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void fill_data(uint8_t a_i[32], const uint8_t *bytes_row_i_column_k, uint8_t size_row_data) noexcept;

}  // namespace sxt::sqcnv
