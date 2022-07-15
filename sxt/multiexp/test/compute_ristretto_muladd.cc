#include "sxt/multiexp/test/compute_ristretto_muladd.h"

#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/type/element_p3.h"

#include "sxt/multiexp/base/exponent_sequence.h"

#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// compute_ristretto_muladd
//--------------------------------------------------------------------------------------------------
void compute_ristretto_muladd(basct::span<rstt::compressed_element> result,
                              basct::span<rstt::compressed_element> generators,
                              basct::span<mtxb::exponent_sequence> sequences) noexcept {

  // compute the expected result
  for (size_t seq = 0; seq < result.size(); ++seq) {
    c21t::element_p3 temp_result = c21cn::zero_p3_v;

    uint8_t element_nbytes = sequences[seq].element_nbytes;

    // sum all the elements in the current sequence together
    for (size_t gen_i = 0; gen_i < sequences[seq].n; ++gen_i) {
      c21t::element_p3 g_i;
      rstb::from_bytes(g_i, generators[gen_i].data());

      basct::cspan<uint8_t> exponent{sequences[seq].data + gen_i * element_nbytes, element_nbytes};

      c21o::scalar_multiply(g_i, exponent, g_i); // h_i = a_i * g_i

      c21o::add(temp_result, temp_result, g_i);
    }

    rstb::to_bytes(result[seq].data(), temp_result);
  }
}
} // namespace sxt::mtxtst
