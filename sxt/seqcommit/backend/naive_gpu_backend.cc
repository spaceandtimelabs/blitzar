#include "sxt/seqcommit/backend/naive_gpu_backend.h"

#include <algorithm>

#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/base/indexed_exponent_sequence.h"
#include "sxt/seqcommit/generator/gpu_generator.h"
#include "sxt/seqcommit/generator/precomputed_generators.h"
#include "sxt/seqcommit/naive/commitment_computation_gpu.h"

namespace sxt::sqcbck {
//--------------------------------------------------------------------------------------------------
// pre_initialize_gpu
//--------------------------------------------------------------------------------------------------
static void pre_initialize_gpu() {
  // initialization of dummy variables
  basct::span<c21t::element_p3> dummy_empty_generators;
  memmg::managed_array<uint8_t> dummy_data_table(1); // 1 col, 1 row, 1 bytes per data
  memmg::managed_array<rstt::compressed_element> dummy_commitments_per_col(1);
  memmg::managed_array<sqcb::indexed_exponent_sequence> dummy_data_cols(1);
  basct::span<rstt::compressed_element> dummy_commitments(dummy_commitments_per_col.data(), 1);
  basct::cspan<sqcb::indexed_exponent_sequence> dummy_value_sequences(dummy_data_cols.data(), 1);

  dummy_data_table[0] = 1;

  auto& data_col = dummy_data_cols[0];

  data_col.indices = nullptr;
  data_col.exponent_sequence.n = 1;
  data_col.exponent_sequence.element_nbytes = 1;
  data_col.exponent_sequence.data = dummy_data_table.data();

  // A small dummy computation to avoid the future cost of JIT compiling PTX code
  sqcnv::compute_commitments_gpu(dummy_commitments, dummy_value_sequences, dummy_empty_generators);
}

//--------------------------------------------------------------------------------------------------
// naive_gpu_backend
//--------------------------------------------------------------------------------------------------
naive_gpu_backend::naive_gpu_backend() { pre_initialize_gpu(); }

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void naive_gpu_backend::compute_commitments(
    basct::span<rstt::compressed_element> commitments,
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
    basct::cspan<c21t::element_p3> generators, uint64_t length_longest_sequence,
    bool has_sparse_sequence) noexcept {

  if (!generators.empty() || has_sparse_sequence) {
    sqcnv::compute_commitments_gpu(commitments, value_sequences, generators);
    return;
  }

  std::vector<c21t::element_p3> generators_data;
  generators = sqcgn::get_precomputed_generators(generators_data, length_longest_sequence, 0, true);

  sqcnv::compute_commitments_gpu(commitments, value_sequences, generators);
}

//--------------------------------------------------------------------------------------------------
// get_generators
//--------------------------------------------------------------------------------------------------
void naive_gpu_backend::get_generators(basct::span<c21t::element_p3> generators,
                                       uint64_t offset_generators) noexcept {
  std::vector<c21t::element_p3> temp_generators_data;
  auto precomputed_generators = sqcgn::get_precomputed_generators(
      temp_generators_data, generators.size(), offset_generators, true);
  std::copy_n(precomputed_generators.begin(), generators.size(), generators.data());
}

//--------------------------------------------------------------------------------------------------
// get_naive_gpu_backend
//--------------------------------------------------------------------------------------------------
naive_gpu_backend* get_naive_gpu_backend() {
  // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  static naive_gpu_backend* backend = new naive_gpu_backend{};
  return backend;
}
} // namespace sxt::sqcbck
