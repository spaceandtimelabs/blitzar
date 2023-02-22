#pragma once

#include <cstddef>
#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::basct {
class blob_array;
}

namespace sxt::mtxmpg {
struct multiproduct_computation_descriptor;

//--------------------------------------------------------------------------------------------------
// setup_multiproduct_computation
//--------------------------------------------------------------------------------------------------
void setup_multiproduct_computation(multiproduct_computation_descriptor& descriptor,
                                    basct::cspan<basct::cspan<uint64_t>> products,
                                    const basct::blob_array& masks, size_t num_inputs) noexcept;
} // namespace sxt::mtxmpg
