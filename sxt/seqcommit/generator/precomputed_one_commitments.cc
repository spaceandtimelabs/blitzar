#include "sxt/seqcommit/generator/precomputed_one_commitments.h"

#include <vector>

#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/cpu_one_commitments.h"

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// precomputed_one_commitments_v
//--------------------------------------------------------------------------------------------------
static basct::cspan<c21t::element_p3> precomputed_one_commitments_v{};

//--------------------------------------------------------------------------------------------------
// init_precomputed_one_commitments
//--------------------------------------------------------------------------------------------------
void init_precomputed_one_commitments(uint64_t n) noexcept {
  if (!precomputed_one_commitments_v.empty() || n == 0) {
    return;
  }

  // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  auto data = new c21t::element_p3[n];

  sqcgn::cpu_get_one_commitments({data, n});

  precomputed_one_commitments_v = {data, n};
}

//--------------------------------------------------------------------------------------------------
// get_precomputed_one_commitments
//--------------------------------------------------------------------------------------------------
basct::cspan<c21t::element_p3> get_precomputed_one_commitments() noexcept {
  return precomputed_one_commitments_v;
}

//--------------------------------------------------------------------------------------------------
// get_precomputed_one_commit
//--------------------------------------------------------------------------------------------------
c21t::element_p3 get_precomputed_one_commit(uint64_t n) noexcept {
  if (precomputed_one_commitments_v.size() > n) {
    return precomputed_one_commitments_v[n];
  }

  if (precomputed_one_commitments_v.empty()) {
    return cpu_get_one_commit(n);
  }

  auto offset = precomputed_one_commitments_v.size() - 1;
  auto remaining_n = n - offset;
  auto prev_commit = precomputed_one_commitments_v[offset];

  return cpu_get_one_commit(prev_commit, remaining_n, offset);
}
} // namespace sxt::sqcgn
