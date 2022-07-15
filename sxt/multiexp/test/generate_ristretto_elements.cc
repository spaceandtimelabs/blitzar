#include "sxt/multiexp/test/generate_ristretto_elements.h"

#include "sxt/base/num/fast_random_number_generator.h"

#include "sxt/curve21/type/element_p3.h"

#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/random/element.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// generate_ristretto_elements
//--------------------------------------------------------------------------------------------------
void generate_ristretto_elements(basct::span<rstt::compressed_element> generators,
                                 std::mt19937& rng) noexcept {

  std::uniform_int_distribution<size_t> generators_gen;
  size_t random_index = generators_gen(rng);
  basn::fast_random_number_generator fast_rng{random_index + 1, random_index + 2};

  // populate the generators array
  for (size_t i = 0; i < generators.size(); ++i) {
    c21t::element_p3 g_i;
    rstrn::generate_random_element(g_i, fast_rng);
    rstb::to_bytes(generators[i].data(), g_i);
  }
}
} // namespace sxt::mtxtst
