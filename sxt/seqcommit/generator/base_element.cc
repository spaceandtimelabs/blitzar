#include "sxt/seqcommit/generator/base_element.h"

#include "sxt/base/num/fast_random_number_generator.h"

#include "sxt/curve21/type/element_p3.h"

#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/random/element.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// compute_base_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void compute_base_element(c21t::element_p3& g, uint64_t index) noexcept {
  // Note: we'll probably substitute a different generator in the future, but
  // this works as a placeholder for now
  basn::fast_random_number_generator rng{index + 1, index + 2};
  rstrn::generate_random_element(g, rng);
}

//--------------------------------------------------------------------------------------------------
// compute_compressed_base_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void compute_compressed_base_element(rstt::compressed_element& g_rt, uint64_t index) noexcept {
  c21t::element_p3 g;
  compute_base_element(g, index);
  rstb::to_bytes(g_rt.data(), g);
}
} // namespace sxt::sqcgn
