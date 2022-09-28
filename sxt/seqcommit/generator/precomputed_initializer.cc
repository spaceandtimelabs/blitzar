#include "sxt/seqcommit/generator/precomputed_initializer.h"

#include "sxt/seqcommit/generator/precomputed_generators.h"
#include "sxt/seqcommit/generator/precomputed_one_commitments.h"

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// init_precomputed_components
//--------------------------------------------------------------------------------------------------
void init_precomputed_components(size_t n, bool use_gpu) noexcept {
  // generators must be initialized before one_commitments as the latter uses the first
  init_precomputed_generators(n, use_gpu);
  init_precomputed_one_commitments(n);
}
} // namespace sxt::sqcgn
