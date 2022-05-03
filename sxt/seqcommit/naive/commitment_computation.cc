#include "sxt/seqcommit/naive/commitment_computation.h"

#include "sxt/seqcommit/base/commitment.h"
#include "sxt/multiexp/base/exponent_sequence.h"

namespace sxt::sqcnv {
//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void compute_commitments(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<mtxb::exponent_sequence> value_sequences) noexcept {
  (void)commitments;
  (void)value_sequences;
}
}  // namespace sxt::sqcnv
