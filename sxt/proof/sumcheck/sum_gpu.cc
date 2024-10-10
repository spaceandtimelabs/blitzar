#include "sxt/proof/sumcheck/sum_gpu.h"

#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum
//--------------------------------------------------------------------------------------------------
void sum(basct::span<s25t::element> polynomial, basdv::stream& stream,
         basct::cspan<s25t::element> mles,
         basct::cspan<std::pair<s25t::element, unsigned>> product_table,
         basct::cspan<unsigned> product_terms, unsigned mid, unsigned n) noexcept {
  (void)polynomial;
  (void)stream;
  (void)mles;
  (void)product_table;
  (void)product_terms;
  (void)mid;
  (void)n;
}
} // namespace sxt::prfsk
