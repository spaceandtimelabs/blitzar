#include "sxt/proof/sumcheck/cpu_driver.h"

#include "sxt/execution/async/future.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// make_workspace
//--------------------------------------------------------------------------------------------------
std::unique_ptr<workspace>
cpu_driver::make_workspace(basct::cspan<s25t::element> mles,
                           basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                           basct::cspan<unsigned> product_terms) const noexcept {
  (void)mles;
  (void)product_table;
  (void)product_terms;
  return {};
}

//--------------------------------------------------------------------------------------------------
// sum
//--------------------------------------------------------------------------------------------------
xena::future<> cpu_driver::sum(basct::span<s25t::element> polynomial,
                             workspace& ws) const noexcept {
  (void)polynomial;
  (void)ws;
  return {};
}

//--------------------------------------------------------------------------------------------------
// fold
//--------------------------------------------------------------------------------------------------
xena::future<> cpu_driver::fold(workspace& ws, const s25t::element& r) const noexcept {
  (void)ws;
  (void)r;
  return {};
}
} // namespace prfsk
