#include "sxt/seqcommit/generator/base_element.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/random/element.h"

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
} // namespace sxt::sqcgn
