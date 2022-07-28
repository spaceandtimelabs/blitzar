#include "sxt/seqcommit/generator/cpu_generator.h"

#include "sxt/base/container/span.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/base_element.h"

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// cpu_get_generators
//--------------------------------------------------------------------------------------------------
void cpu_get_generators(basct::span<c21t::element_p3> generators, uint64_t offset) noexcept {
  for (uint64_t index = 0; index < generators.size(); ++index) {
    sqcgn::compute_base_element(generators[index], index + offset);
  }
}
} // namespace sxt::sqcgn
