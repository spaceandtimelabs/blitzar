#pragma once

#include "sxt/base/container/span.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::mtxmpg {
struct multiproduct_computation_descriptor;

//--------------------------------------------------------------------------------------------------
// setup_multiproduct_computation
void setup_multiproduct_computation(multiproduct_computation_descriptor& descriptor,
                                    memmg::managed_array<unsigned>&& indexes,
                                    basct::cspan<unsigned> product_sizes) noexcept;
} // namespace sxt::mtxmpg
