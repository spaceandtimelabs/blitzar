#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// multiexponentiate_options 
//--------------------------------------------------------------------------------------------------
struct multiexponentiate_options {
  unsigned min_chunk_size = 1'000u;
  unsigned max_chunk_size = 256'000u;
  unsigned bit_width = 8;
};

//--------------------------------------------------------------------------------------------------
// plan_multiexponentiation 
//--------------------------------------------------------------------------------------------------
void plan_multiexponentiation(multiexponentiate_options& options, unsigned num_outputs,
                              unsigned element_num_bytes, unsigned n) noexcept;

//--------------------------------------------------------------------------------------------------
// multiexponentiate2 
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
xena::future<>
multiexponentiate2(basct::span<Element> res, const multiexponentiate_options& options,
                   basct::cspan<Element> generators, basct::cspan<const Element*> scalars,
                   unsigned element_num_bytes, unsigned n) noexcept {
  (void)res;
  (void)options;
  (void)scalars;
  (void)element_num_bytes;
  (void)n;
  return {};
}

//--------------------------------------------------------------------------------------------------
// try_multiexponentiate2
//--------------------------------------------------------------------------------------------------
/**
 * Attempt to compute a multi-exponentiation using the bucket method if the problem dimensions
 * suggest it will give a performance benefit; otherwise, return an empty array.
 */
template <bascrv::element Element>
xena::future<memmg::managed_array<Element>>
try_multiexponentiate2(basct::cspan<Element> generators,
                      basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  auto num_outputs = exponents.size();
  if (num_outputs == 0) {
    co_return {};
  }
  auto n = exponents[0].n;
  if (n == 0) {
    co_return {};
  }
  memmg::managed_array<const uint8_t*> scalars(num_outputs);
  scalars[0] = exponents[0].data;
  auto element_num_bytes = exponents[0].element_nbytes;
  for (size_t output_index=1; output_index<num_outputs; ++output_index) {
    auto& seq = exponents[output_index];
    if (seq.n != n || seq.element_nbytes != element_num_bytes) {
      co_return {};
    }
    scalars[output_index] = seq.data;
  }
  memmg::managed_array<Element> res{num_outputs, memr::get_pinned_resource()};
  multiexponentiate_options options;
  plan_multiexponentiation(options, num_outputs, element_num_bytes, n);
  co_await multiexponentiate2(res, options, generators, scalars, element_num_bytes, n);
  co_return res;
}
} // namespace sxt::mtxbk