#pragma once

#include <cstddef>

namespace sxt::mtxpmp {
struct multiproduct_params;

//--------------------------------------------------------------------------------------------------
// compute_multiproduct_params
//--------------------------------------------------------------------------------------------------
void compute_multiproduct_params(multiproduct_params& params, size_t num_outputs,
                                 size_t num_inputs) noexcept;
} // namespace sxt::mtxpmp
