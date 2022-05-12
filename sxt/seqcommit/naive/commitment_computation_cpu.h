#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxb { struct exponent_sequence; }
namespace sxt::sqcb { class commitment; }

namespace sxt::sqcnv {

//--------------------------------------------------------------------------------------------------
// compute_commitments_cpu
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void compute_commitments_cpu(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<mtxb::exponent_sequence> value_sequences) noexcept;

}  // namespace sxt::sqcnv
