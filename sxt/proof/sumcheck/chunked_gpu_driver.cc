#include "sxt/proof/sumcheck/chunked_gpu_driver.h"

#include <algorithm>

#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// chunked_gpu_workspace
//--------------------------------------------------------------------------------------------------
namespace {
struct chunked_gpu_workspace final : public workspace {
  basct::cspan<s25t::element> mles0;
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
xena::future<std::unique_ptr<workspace>>
chunked_gpu_driver::make_workspace(basct::cspan<s25t::element> mles,
                                   basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                                   basct::cspan<unsigned> product_terms,
                                   unsigned n) const noexcept {
  auto res = std::make_unique<chunked_gpu_workspace>();
  res->mles0 = mles;
  res->product_table =product_table;
  res->product_terms = product_terms;
  res->n = n;
  res->num_variables = std::max(basn::ceil_log2(n), 1);
  return xena::make_ready_future<std::unique_ptr<workspace>>(std::move(res));
}

//--------------------------------------------------------------------------------------------------
// sum
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
xena::future<> chunked_gpu_driver::sum(basct::span<s25t::element> polynomial,
                                       workspace& ws) const noexcept {
  return {};
}
#pragma clang diagnostic pop

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
