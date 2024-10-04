#include "sxt/proof/sumcheck/cpu_driver.h"

#include <algorithm>

#include "sxt/base/container/stack_array.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
#include "sxt/scalar25/operation/muladd.h"
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
  auto& work = static_cast<cpu_workspace&>(ws);
  auto n = work.n;
  auto mid = 1u << (work.num_variables - 1u);
  SXT_RELEASE_ASSERT(work.n > mid);

  auto mles = work.mles.data();
  auto product_table = work.product_table;
  auto product_terms = work.product_terms;

  for (auto& val : polynomial) {
    val = {};
  }

  auto n1 = work.n - mid;
  for (unsigned i = 0; i < n1; ++i) {
    unsigned term_first = 0;
    for (auto [mult, num_terms] : product_table) {
      auto terms = product_terms.subspan(term_first, num_terms);
      SXT_STACK_ARRAY(p, num_terms, s25t::element);
      expand_products(p, mles + i, n, mid, terms);
      for (unsigned term_index=0; term_index<num_terms; ++term_index) {
        s25o::muladd(polynomial[term_index], mult, p[term_index], polynomial[term_index]);
      }
      term_first += num_terms;
    }
  }

  auto n2 = mid - n1;
  (void)n2;
  (void)mid;
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
