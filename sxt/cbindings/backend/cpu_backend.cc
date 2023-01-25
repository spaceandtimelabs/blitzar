#include "sxt/cbindings/backend/cpu_backend.h"

#include <cstring>
#include <vector>

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
#include "sxt/seqcommit/generator/precomputed_generators.h"

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void cpu_backend::compute_commitments(basct::span<rstt::compressed_element> commitments,
                                      basct::cspan<mtxb::exponent_sequence> value_sequences,
                                      basct::cspan<c21t::element_p3> generators) const noexcept {
  memmg::managed_array<rstt::compressed_element> inout;
  mtxrs::precomputed_p3_input_accessor input_accessor{generators};
  mtxrs::pippenger_multiproduct_solver multiproduct_solver;
  mtxrs::multiexponentiation_cpu_driver drv{&input_accessor, &multiproduct_solver};
  mtxpi::compute_multiexponentiation(inout, drv, value_sequences);
  std::memcpy(commitments.data(), inout.data(),
              commitments.size() * sizeof(rstt::compressed_element));
}

//--------------------------------------------------------------------------------------------------
// get_precomputed_generators
//--------------------------------------------------------------------------------------------------
basct::cspan<c21t::element_p3>
cpu_backend::get_precomputed_generators(std::vector<c21t::element_p3>& temp_generators, uint64_t n,
                                        uint64_t offset_generators) const noexcept {
  return sqcgn::get_precomputed_generators(temp_generators, n, offset_generators, false);
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
