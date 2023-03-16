#pragma once

#include "sxt/base/container/span.h"

namespace sxt::mtxmpg {
struct multiproduct_computation_descriptor;

//--------------------------------------------------------------------------------------------------
// setup_multiproduct_computation
//--------------------------------------------------------------------------------------------------
void setup_multiproduct_computation(multiproduct_computation_descriptor& descriptor,
                                    basct::cspan<unsigned> product_sizes) noexcept;
} // namespace sxt::mtxmpg
