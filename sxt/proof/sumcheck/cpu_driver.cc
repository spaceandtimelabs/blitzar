#include "sxt/proof/sumcheck/cpu_driver.h"

#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// cpu_workspace
//--------------------------------------------------------------------------------------------------
namespace {
struct cpu_workspace final : public workspace {
  memmg::managed_array<s25t::element> mles;
  basct::cspan<std::pair<s25t::element, unsigned>> product_table;
  basct::cspan<unsigned> product_terms;
  unsigned n;
  unsigned num_variables;
};
} // namespace

//--------------------------------------------------------------------------------------------------
// make_workspace
//--------------------------------------------------------------------------------------------------
std::unique_ptr<workspace>
cpu_driver::make_workspace(basct::cspan<s25t::element> mles,
                           basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                           basct::cspan<unsigned> product_terms, unsigned n) const noexcept {
  auto res = std::make_unique<cpu_workspace>();
  res->mles = memmg::managed_array<s25t::element>{mles.begin(), mles.end()};
  res->product_table = product_table;
  res->product_terms = product_terms;
  res->n = n;
  res->num_variables = basn::ceil_log2(n);
  return res;
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
