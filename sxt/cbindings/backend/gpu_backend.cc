#include "sxt/cbindings/backend/gpu_backend.h"

#include <vector>

#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/proof/inner_product/cpu_driver.h"
#include "sxt/proof/inner_product/proof_computation.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/seqcommit/generator/precomputed_generators.h"
#include "sxt/seqcommit/naive/commitment_computation_gpu.h"

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// pre_initialize_gpu
//--------------------------------------------------------------------------------------------------
static void pre_initialize_gpu() {
  // initialization of dummy variables
  memmg::managed_array<c21t::element_p3> dummy_empty_generators(1);
  memmg::managed_array<uint8_t> dummy_data_table(1); // 1 col, 1 row, 1 bytes per data
  memmg::managed_array<rstt::compressed_element> dummy_commitments_per_col(1);
  memmg::managed_array<mtxb::exponent_sequence> dummy_data_cols(1);
  basct::span<rstt::compressed_element> dummy_commitments(dummy_commitments_per_col.data(), 1);
  basct::cspan<mtxb::exponent_sequence> dummy_value_sequences(dummy_data_cols.data(), 1);

  dummy_data_table[0] = 1;

  auto& data_col = dummy_data_cols[0];

  data_col.n = 1;
  data_col.element_nbytes = 1;
  data_col.data = dummy_data_table.data();

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
                                      basct::cspan<mtxb::exponent_sequence> value_sequences,
                                      basct::cspan<c21t::element_p3> generators) const noexcept {
  sqcnv::compute_commitments_gpu(commitments, value_sequences, generators);
}

//--------------------------------------------------------------------------------------------------
// get_precomputed_generators
//--------------------------------------------------------------------------------------------------
basct::cspan<c21t::element_p3>
gpu_backend::get_precomputed_generators(std::vector<c21t::element_p3>& temp_generators, uint64_t n,
                                        uint64_t offset_generators) const noexcept {
  return sqcgn::get_precomputed_generators(temp_generators, n, offset_generators, true);
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
