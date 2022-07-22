#include "sxt/multiexp/test/generate_curve21_elements.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/random/element.h"

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// generate_curve21_elements
//--------------------------------------------------------------------------------------------------
void generate_curve21_elements(basct::span<c21t::element_p3> generators,
                               std::mt19937& rng) noexcept {

  std::uniform_int_distribution<size_t> generators_gen;
  size_t random_index = generators_gen(rng);
  basn::fast_random_number_generator fast_rng{random_index + 1, random_index + 2};

  // populate the generators array
  for (size_t i = 0; i < generators.size(); ++i) {
    rstrn::generate_random_element(generators[i], fast_rng);
  }
}
} // namespace sxt::mtxtst
