#include "sxt/proof/sumcheck/chunked_gpu_driver.h"

#include <algorithm>

#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/proof/sumcheck/device_cache.h"
#include "sxt/proof/sumcheck/sum_gpu.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// chunked_gpu_workspace
//--------------------------------------------------------------------------------------------------
namespace {
struct chunked_gpu_workspace final : public workspace {
  device_cache cache;
  basct::cspan<s25t::element> mles0;
  memmg::managed_array<s25t::element> mles;
  unsigned n;
  unsigned num_variables;

  chunked_gpu_workspace(basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                        basct::cspan<unsigned> product_terms) noexcept
      : cache{product_table, product_terms} {}
};
} // namespace

//--------------------------------------------------------------------------------------------------
// make_workspace
//--------------------------------------------------------------------------------------------------
xena::future<std::unique_ptr<workspace>>
chunked_gpu_driver::make_workspace(basct::cspan<s25t::element> mles,
                                   basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                                   basct::cspan<unsigned> product_terms,
                                   unsigned n) const noexcept {
  auto res = std::make_unique<chunked_gpu_workspace>(product_table, product_terms);
  res->mles0 = mles;
  res->n = n;
  res->num_variables = std::max(basn::ceil_log2(n), 1);
  return xena::make_ready_future<std::unique_ptr<workspace>>(std::move(res));
}

//--------------------------------------------------------------------------------------------------
// sum
//--------------------------------------------------------------------------------------------------
xena::future<> chunked_gpu_driver::sum(basct::span<s25t::element> polynomial,
                                       workspace& ws) const noexcept {
  auto& work = static_cast<chunked_gpu_workspace&>(ws);
  basct::cspan<s25t::element> mles = work.mles.empty() ? work.mles0 : work.mles;
  co_await sum_gpu(polynomial, work.cache, mles, work.n);
}

//--------------------------------------------------------------------------------------------------
// fold
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
xena::future<> chunked_gpu_driver::fold(workspace& ws, const s25t::element& r) const noexcept {
  return {};
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
