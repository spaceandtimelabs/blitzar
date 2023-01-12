#include "sxt/cbindings/backend/cpu_backend.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/pippenger/multiexponentiation.h"
#include "sxt/multiexp/ristretto/multiexponentiation_cpu_driver.h"
#include "sxt/multiexp/ristretto/pippenger_multiproduct_solver.h"
#include "sxt/multiexp/ristretto/precomputed_p3_input_accessor.h"
#include "sxt/proof/inner_product/cpu_driver.h"
#include "sxt/proof/inner_product/proof_computation.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/seqcommit/base/indexed_exponent_sequence.h"
#include "sxt/seqcommit/generator/cpu_generator.h"
#include "sxt/seqcommit/generator/precomputed_generators.h"
#include "sxt/seqcommit/naive/commitment_computation_cpu.h"

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// populate_exponents_array
//--------------------------------------------------------------------------------------------------
// returns true if there is some sparse sequence in
// the `value_sequences` span; otherwise returns false
static void
populate_exponents_array(memmg::managed_array<mtxb::exponent_sequence>& exponents,
                         basct::cspan<sqcb::indexed_exponent_sequence> value_sequences) {

  for (size_t i = 0; i < value_sequences.size(); ++i) {
    exponents[i] = value_sequences[i].exponent_sequence;
  }
}

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void cpu_backend::compute_commitments(basct::span<rstt::compressed_element> commitments,
                                      basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
                                      basct::cspan<c21t::element_p3> generators,
                                      uint64_t length_longest_sequence,
                                      bool has_sparse_sequence) const noexcept {

  memmg::managed_array<mtxb::exponent_sequence> exponents(value_sequences.size());

  populate_exponents_array(exponents, value_sequences);

  if (has_sparse_sequence) {
    /////////////////////////////////////////////////////////
    // TODO
    /////////////////////////////////////////////////////////
    // for now, we use the naive cpu implementation
    // to process sparse sequences. But later, this should
    // be changed to use the pippenger implementation instead
    /////////////////////////////////////////////////////////
    return sqcnv::compute_commitments_cpu(commitments, value_sequences, generators);
  }
  memmg::managed_array<rstt::compressed_element> inout;
  std::vector<c21t::element_p3> generators_data;
  if (generators.empty()) {
    generators =
        sqcgn::get_precomputed_generators(generators_data, length_longest_sequence, 0, false);
  }
  mtxrs::precomputed_p3_input_accessor input_accessor{generators};
  mtxrs::pippenger_multiproduct_solver multiproduct_solver;
  mtxrs::multiexponentiation_cpu_driver drv{&input_accessor, &multiproduct_solver};
  mtxpi::compute_multiexponentiation(inout, drv, exponents);
  std::memcpy(commitments.data(), inout.data(),
              commitments.size() * sizeof(rstt::compressed_element));
}

//--------------------------------------------------------------------------------------------------
// get_generators
//--------------------------------------------------------------------------------------------------
void cpu_backend::get_generators(basct::span<c21t::element_p3> generators,
                                 uint64_t offset_generators) const noexcept {
  std::vector<c21t::element_p3> temp_generators_data;
  auto precomputed_generators = sqcgn::get_precomputed_generators(
      temp_generators_data, generators.size(), offset_generators, false);
  std::copy_n(precomputed_generators.begin(), generators.size(), generators.data());
}

//--------------------------------------------------------------------------------------------------
// prove_inner_product
//--------------------------------------------------------------------------------------------------
void cpu_backend::prove_inner_product(basct::span<rstt::compressed_element> l_vector,
                                      basct::span<rstt::compressed_element> r_vector,
                                      s25t::element& ap_value, prft::transcript& transcript,
                                      const prfip::proof_descriptor& descriptor,
                                      basct::cspan<s25t::element> a_vector) const noexcept {
  prfip::cpu_driver drv;
  prfip::prove_inner_product(l_vector, r_vector, ap_value, transcript, drv, descriptor, a_vector);
}

//--------------------------------------------------------------------------------------------------
// verify_inner_product
//--------------------------------------------------------------------------------------------------
bool cpu_backend::verify_inner_product(prft::transcript& transcript,
                                       const prfip::proof_descriptor& descriptor,
                                       const s25t::element& product,
                                       const c21t::element_p3& a_commit,
                                       basct::cspan<rstt::compressed_element> l_vector,
                                       basct::cspan<rstt::compressed_element> r_vector,
                                       const s25t::element& ap_value) const noexcept {
  prfip::cpu_driver drv;
  return prfip::verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector,
                                     r_vector, ap_value);
}

//--------------------------------------------------------------------------------------------------
// get_cpu_backend
//--------------------------------------------------------------------------------------------------
cpu_backend* get_cpu_backend() {
  // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  static cpu_backend* backend = new cpu_backend{};
  return backend;
}
} // namespace sxt::cbnbck
