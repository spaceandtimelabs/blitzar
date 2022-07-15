#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::sqcb {
struct indexed_exponent_sequence;
}
namespace sxt::rstt {
class compressed_element;
}

namespace sxt::sqcnv {

//--------------------------------------------------------------------------------------------------
// compute_commitments_cpu
//--------------------------------------------------------------------------------------------------
void compute_commitments_cpu(basct::span<rstt::compressed_element> commitments,
                             basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
                             basct::cspan<rstt::compressed_element> generators) noexcept;

} // namespace sxt::sqcnv
