#include "sxt/cbindings/backend/gpu_backend.h"

#include <algorithm>

#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/proof/inner_product/cpu_driver.h"
#include "sxt/proof/inner_product/proof_computation.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/seqcommit/base/indexed_exponent_sequence.h"
#include "sxt/seqcommit/generator/gpu_generator.h"
#include "sxt/seqcommit/generator/precomputed_generators.h"
#include "sxt/seqcommit/naive/commitment_computation_gpu.h"

namespace sxt::cbnbck {
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
// gpu_backend
//--------------------------------------------------------------------------------------------------
gpu_backend::gpu_backend() { pre_initialize_gpu(); }

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void gpu_backend::compute_commitments(basct::span<rstt::compressed_element> commitments,
                                      basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
                                      basct::cspan<c21t::element_p3> generators,
                                      uint64_t length_longest_sequence,
                                      bool has_sparse_sequence) const noexcept {

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
void gpu_backend::get_generators(basct::span<c21t::element_p3> generators,
                                 uint64_t offset_generators) const noexcept {
  std::vector<c21t::element_p3> temp_generators_data;
  auto precomputed_generators = sqcgn::get_precomputed_generators(
      temp_generators_data, generators.size(), offset_generators, true);
  std::copy_n(precomputed_generators.begin(), generators.size(), generators.data());
}

//--------------------------------------------------------------------------------------------------
// prove_inner_product
//--------------------------------------------------------------------------------------------------
void gpu_backend::prove_inner_product(basct::span<rstt::compressed_element> l_vector,
                                      basct::span<rstt::compressed_element> r_vector,
                                      s25t::element& ap_value, prft::transcript& transcript,
                                      const prfip::proof_descriptor& descriptor,
                                      basct::cspan<s25t::element> a_vector) const noexcept {
  // TODO: update this to use gpu_driver when available
  prfip::cpu_driver drv;
  prfip::prove_inner_product(l_vector, r_vector, ap_value, transcript, drv, descriptor, a_vector);
}

//--------------------------------------------------------------------------------------------------
// verify_inner_product
//--------------------------------------------------------------------------------------------------
bool gpu_backend::verify_inner_product(prft::transcript& transcript,
                                       const prfip::proof_descriptor& descriptor,
                                       const s25t::element& product,
                                       const c21t::element_p3& a_commit,
                                       basct::cspan<rstt::compressed_element> l_vector,
                                       basct::cspan<rstt::compressed_element> r_vector,
                                       const s25t::element& ap_value) const noexcept {
  // TODO: update this to use gpu_driver when available
  prfip::cpu_driver drv;
  return prfip::verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector,
                                     r_vector, ap_value);
}

//--------------------------------------------------------------------------------------------------
// get_gpu_backend
//--------------------------------------------------------------------------------------------------
gpu_backend* get_gpu_backend() {
  // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  static gpu_backend* backend = new gpu_backend{};
  return backend;
}
} // namespace sxt::cbnbck
