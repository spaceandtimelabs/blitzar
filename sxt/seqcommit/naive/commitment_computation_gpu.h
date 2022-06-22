#pragma once

#include <cinttypes>

#include "sxt/base/container/span.h"

namespace sxt::sqcb { struct indexed_exponent_sequence; }
namespace sxt::sqcb { class commitment; }
namespace sxt::c21t { struct element_p3; }

namespace sxt::sqcnv {

//--------------------------------------------------------------------------------------------------
// compute_commitments_gpu
//--------------------------------------------------------------------------------------------------
void compute_commitments_gpu(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
    basct::cspan<sqcb::commitment> generators
) noexcept;

} // namespace sxt
