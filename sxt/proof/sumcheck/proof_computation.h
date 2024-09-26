#pragma once

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future.h"
/* #include "sxt/proof/transcript/transcript_utility.h" */

namespace sxt::prft { class transcript; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// prove_sum 
//--------------------------------------------------------------------------------------------------
template <class Scalar>
xena::future<> prove_sum(basct::span<Scalar> polynomials, prft::transcript& transcript,
                         basct::cspan<Scalar> mles, basct::cspan<unsigned> product_table,
                         basct::cspan<unsigned> product_lengths) noexcept {
  (void)polynomials;
  (void)transcript;
  (void)mles;
  (void)product_table;
  (void)product_lengths;
  return {};
}
} // namespace sxt::prfsk
