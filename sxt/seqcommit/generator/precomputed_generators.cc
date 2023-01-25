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
get_precomputed_generators(std::vector<c21t::element_p3>& generators_data, size_t length_generators,
                           size_t offset, bool use_gpu) noexcept {
  if (precomputed_generators_v.size() >= length_generators + offset) {
    return precomputed_generators_v.subspan(offset, length_generators);
  }

  generators_data.resize(length_generators);

  size_t gen_span_offset = 0;
  basct::span<c21t::element_p3> gen_span;

  if (precomputed_generators_v.size() > offset) {
    std::copy(precomputed_generators_v.begin() + offset, precomputed_generators_v.end(),
              generators_data.begin());

    // compute generators from `gen_span_offset...(gen_span.size() + gen_span_offset)`
    gen_span_offset = precomputed_generators_v.size();

    gen_span = {generators_data.data() + (precomputed_generators_v.size() - offset),
                (length_generators + offset) - precomputed_generators_v.size()};
  } else {
    // compute generators from `offset...(offset + length_generators)`
    gen_span_offset = offset;

    gen_span = {generators_data.data(), length_generators};
  }

  if (use_gpu) {
    sqcgn::gpu_get_generators(gen_span, gen_span_offset);
  } else {
    sqcgn::cpu_get_generators(gen_span, gen_span_offset);
  }

  return generators_data;
}
} // namespace sxt::sqcgn
