#pragma once

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
    return {};
  }

  xena::future<> fold(workspace& ws, T& r) const noexcept override {
    return {};
  }
};
} // namespace sxt::prfsk2
