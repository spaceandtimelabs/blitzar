#include "sxt/seqcommit/generator/cpu_one_commitments.h"

#include <vector>

#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/precomputed_generators.h"

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// cpu_get_one_commitments
//--------------------------------------------------------------------------------------------------
void cpu_get_one_commitments(basct::span<c21t::element_p3> one_commitments) noexcept {
  auto prev_commit = c21cn::zero_p3_v;

  auto n = one_commitments.size();
  std::vector<c21t::element_p3> generators_data;
  auto precomputed_gens = get_precomputed_generators(generators_data, n, 0, false);

  for (uint64_t i = 0; i < n; ++i) {
    one_commitments[i] = prev_commit;
    c21o::add(prev_commit, prev_commit, precomputed_gens[i]);
  }
}

//--------------------------------------------------------------------------------------------------
// cpu_get_one_commit
//--------------------------------------------------------------------------------------------------
c21t::element_p3 cpu_get_one_commit(c21t::element_p3 prev_commit, uint64_t n,
                                    uint64_t offset) noexcept {
  std::vector<c21t::element_p3> generators_data;
  auto precomputed_gens = get_precomputed_generators(generators_data, n, offset, false);

  for (uint64_t i = 0; i < n; ++i) {
    c21o::add(prev_commit, prev_commit, precomputed_gens[i]);
  }

  return prev_commit;
}

c21t::element_p3 cpu_get_one_commit(uint64_t n) noexcept {
  return cpu_get_one_commit(c21cn::zero_p3_v, n, 0);
}
} // namespace sxt::sqcgn
