#include "sxt/seqcommit/backend/naive_cpu_backend.h"

#include <vector>

#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/base/indexed_exponent_sequence.h"
#include "sxt/seqcommit/generator/cpu_generator.h"
#include "sxt/seqcommit/generator/precomputed_generators.h"
#include "sxt/seqcommit/naive/commitment_computation_cpu.h"

namespace sxt::sqcbck {

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void naive_cpu_backend::compute_commitments(
    basct::span<rstt::compressed_element> commitments,
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
    basct::cspan<c21t::element_p3> generators, uint64_t length_longest_sequence,
    bool has_sparse_sequence) noexcept {

  if (!generators.empty() || has_sparse_sequence) {
    sqcnv::compute_commitments_cpu(commitments, value_sequences, generators);
    return;
  }

  std::vector<c21t::element_p3> generators_data;

  generators = sqcgn::get_precomputed_generators(generators_data, length_longest_sequence, false);

  sqcnv::compute_commitments_cpu(commitments, value_sequences, generators);
}

//--------------------------------------------------------------------------------------------------
// get_generators
//--------------------------------------------------------------------------------------------------
void naive_cpu_backend::get_generators(basct::span<c21t::element_p3> generators,
                                       uint64_t offset_generators) noexcept {
  sqcgn::cpu_get_generators(generators, offset_generators);
}

//--------------------------------------------------------------------------------------------------
// get_naive_cpu_backend
//--------------------------------------------------------------------------------------------------
naive_cpu_backend* get_naive_cpu_backend() {
  // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  static naive_cpu_backend* backend = new naive_cpu_backend{};
  return backend;
}
} // namespace sxt::sqcbck
