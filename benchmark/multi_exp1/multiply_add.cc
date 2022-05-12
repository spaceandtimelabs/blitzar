#include "benchmark/multi_exp1/multiply_add.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/random/exponent.h"
#include "sxt/curve21/random/ristretto_element.h"
#include "sxt/curve21/type/element_p3.h"

namespace sxt {
//--------------------------------------------------------------------------------------------------
// multiply_add
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void multiply_add(c21t::element_p3& res, int mi, int i) noexcept {
  basn::fast_random_number_generator rng{static_cast<uint64_t>(i + 1),
                                         static_cast<uint64_t>(mi + 1)};

  // pretend like g is a random element rather than fixed
  c21t::element_p3 g;
  c21rn::generate_random_ristretto_element(g, rng);

  unsigned char a[32];
  c21rn::generate_random_exponent(a, rng);
  c21t::element_p3 e;
  c21o::scalar_multiply255(e, a, g);
  c21o::add(res, res, e);
}
} // namespace sxt
