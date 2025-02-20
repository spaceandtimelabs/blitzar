#pragma once

#include "sxt/base/error/assert.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/proof/sumcheck2/driver.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/memory/management/managed_array.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// cpu_driver
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
class cpu_driver final : public driver<T> {
  struct cpu_workspace final : public workspace {
    memmg::managed_array<T> mles;
    basct::cspan<std::pair<T, unsigned>> product_table;
    basct::cspan<unsigned> product_terms;
    unsigned n;
    unsigned num_variables;
  };

public:
  // driver
  xena::future<std::unique_ptr<workspace>>
  make_workspace(basct::cspan<T> mles,
                 basct::cspan<std::pair<T, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, unsigned n) const noexcept override {
    auto res = std::make_unique<cpu_workspace>();
    res->mles = memmg::managed_array<T>{mles.begin(), mles.end()};
    res->product_table = product_table;
    res->product_terms = product_terms;
    res->n = n;
    res->num_variables = std::max(basn::ceil_log2(n), 1);
    return xena::make_ready_future<std::unique_ptr<workspace>>(std::move(res));
  }

  xena::future<> sum(basct::span<T> polynomial, workspace& ws) const noexcept override {
  auto& work = static_cast<cpu_workspace&>(ws);
  auto n = work.n;
  auto mid = 1u << (work.num_variables - 1u);
  SXT_RELEASE_ASSERT(work.n >= mid);

  auto mles = work.mles.data();
  auto product_table = work.product_table;
  auto product_terms = work.product_terms;

  for (auto& val : polynomial) {
    val = {};
  }

  // expand paired terms
#if 0
  auto n1 = work.n - mid;
  for (unsigned i = 0; i < n1; ++i) {
    unsigned term_first = 0;
    for (auto [mult, num_terms] : product_table) {
      SXT_RELEASE_ASSERT(num_terms < polynomial.size());
      auto terms = product_terms.subspan(term_first, num_terms);
      SXT_STACK_ARRAY(p, num_terms + 1u, T);
      expand_products(p, mles + i, n, mid, terms);
      for (unsigned term_index = 0; term_index < p.size(); ++term_index) {
        muladd(polynomial[term_index], mult, p[term_index], polynomial[term_index]);
      }
      term_first += num_terms;
    }
  }

  // expand terms where the corresponding pair is zero (i.e. n is not a power of 2)
  for (unsigned i = n1; i < mid; ++i) {
    unsigned term_first = 0;
    for (auto [mult, num_terms] : product_table) {
      auto terms = product_terms.subspan(term_first, num_terms);
      SXT_STACK_ARRAY(p, num_terms + 1u, T);
      partial_expand_products(p, mles + i, n, terms);
      for (unsigned term_index = 0; term_index < p.size(); ++term_index) {
        muladd(polynomial[term_index], mult, p[term_index], polynomial[term_index]);
      }
      term_first += num_terms;
    }
  }
#endif

  return xena::make_ready_future();
  }

  xena::future<> fold(workspace& ws, T& r) const noexcept override {
    return {};
  }
};
} // namespace sxt::prfsk2
