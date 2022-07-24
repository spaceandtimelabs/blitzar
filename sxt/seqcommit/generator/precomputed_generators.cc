#include "sxt/seqcommit/generator/precomputed_generators.h"

#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/base_element.h"

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// precomputed_generators_v
//--------------------------------------------------------------------------------------------------
static basct::cspan<c21t::element_p3> precomputed_generators_v{};

//--------------------------------------------------------------------------------------------------
// init_precomputed_generators
//--------------------------------------------------------------------------------------------------
void init_precomputed_generators(size_t n) noexcept {
  if (!precomputed_generators_v.empty() || n == 0) {
    return;
  }

  // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  auto data = new c21t::element_p3[n];
  for (size_t index = 0; index < n; ++index) {
    compute_base_element(data[index], index);
  }
  precomputed_generators_v = {data, n};
}

//--------------------------------------------------------------------------------------------------
// get_precomputed_generators
//--------------------------------------------------------------------------------------------------
basct::cspan<c21t::element_p3> get_precomputed_generators() noexcept {
  return precomputed_generators_v;
}
} // namespace sxt::sqcgn
