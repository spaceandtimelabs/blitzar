#pragma once

#include "sxt/base/container/span.h"

namespace sxt::c21t { struct element_p3; }
namespace sxt::mtxb { struct exponent_sequence; }
namespace sxt::sqcb { class commitment; }

namespace sxt::sqcnv {
//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void compute_commitments(
    basct::span<sqcb::commitment> &commitments,
    const basct::cspan<mtxb::exponent_sequence> &value_sequences) noexcept;
}  // namespace sxt::sqcnv
