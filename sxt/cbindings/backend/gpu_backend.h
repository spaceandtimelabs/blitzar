#pragma once

#include <cinttypes>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/cbindings/backend/computational_backend.h"

namespace sxt::mtxb {
struct exponent_sequence;
}
namespace sxt::rstt {
class compressed_element;
}
namespace sxt::c21t {
struct element_p3;
}

namespace sxt::s25t {
struct element;
}

namespace sxt::prft {
class transcript;
}

namespace sxt::prfip {
struct proof_descriptor;
}

namespace sxt::cbnbck {

//--------------------------------------------------------------------------------------------------
// gpu_backend
//--------------------------------------------------------------------------------------------------
class gpu_backend final : public computational_backend {
public:
  gpu_backend();

  void compute_commitments(basct::span<rstt::compressed_element> commitments,
                           basct::cspan<mtxb::exponent_sequence> value_sequences,
                           basct::cspan<c21t::element_p3> generators) const noexcept override;

  basct::cspan<c21t::element_p3>
  get_precomputed_generators(std::vector<c21t::element_p3>& temp_generators, uint64_t n,
                             uint64_t offset_generators) const noexcept override;

  void prove_inner_product(basct::span<rstt::compressed_element> l_vector,
                           basct::span<rstt::compressed_element> r_vector, s25t::element& ap_value,
                           prft::transcript& transcript, const prfip::proof_descriptor& descriptor,
                           basct::cspan<s25t::element> a_vector) const noexcept override;

  bool verify_inner_product(prft::transcript& transcript, const prfip::proof_descriptor& descriptor,
                            const s25t::element& product, const c21t::element_p3& a_commit,
                            basct::cspan<rstt::compressed_element> l_vector,
                            basct::cspan<rstt::compressed_element> r_vector,
                            const s25t::element& ap_value) const noexcept override;
};

//--------------------------------------------------------------------------------------------------
// get_gpu_backend
//--------------------------------------------------------------------------------------------------
gpu_backend* get_gpu_backend();

} // namespace sxt::cbnbck
