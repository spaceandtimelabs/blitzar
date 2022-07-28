#include "sxt/seqcommit/generator/precomputed_generators.h"

#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/cpu_generator.h"
#include "sxt/seqcommit/generator/gpu_generator.h"

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// precomputed_generators_v
//--------------------------------------------------------------------------------------------------
static basct::cspan<c21t::element_p3> precomputed_generators_v{};

//--------------------------------------------------------------------------------------------------
// init_precomputed_generators
//--------------------------------------------------------------------------------------------------
void init_precomputed_generators(size_t n, bool use_gpu) noexcept {
  if (!precomputed_generators_v.empty() || n == 0) {
    return;
  }

  // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  auto data = new c21t::element_p3[n];

  if (use_gpu) {
    sqcgn::gpu_get_generators({data, n}, 0);
  } else {
    sqcgn::cpu_get_generators({data, n}, 0);
  }

  precomputed_generators_v = {data, n};
}

//--------------------------------------------------------------------------------------------------
// get_precomputed_generators
//--------------------------------------------------------------------------------------------------
basct::cspan<c21t::element_p3> get_precomputed_generators() noexcept {
  return precomputed_generators_v;
}

basct::cspan<c21t::element_p3>
get_precomputed_generators(std::vector<c21t::element_p3>& generators_data,
                           size_t length_longest_sequence, bool use_gpu) noexcept {

  if (precomputed_generators_v.size() >= length_longest_sequence) {
    return precomputed_generators_v;
  }

  generators_data.resize(length_longest_sequence);

  std::copy(precomputed_generators_v.begin(), precomputed_generators_v.end(),
            generators_data.begin());

  if (use_gpu) {
    sqcgn::gpu_get_generators(
        basct::span<c21t::element_p3>{generators_data.data() + precomputed_generators_v.size(),
                                      length_longest_sequence - precomputed_generators_v.size()},
        precomputed_generators_v.size());
  } else {
    sqcgn::cpu_get_generators(
        basct::span<c21t::element_p3>{generators_data.data() + precomputed_generators_v.size(),
                                      length_longest_sequence - precomputed_generators_v.size()},
        precomputed_generators_v.size());
  }

  return generators_data;
}
} // namespace sxt::sqcgn
