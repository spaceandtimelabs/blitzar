#include "sxt/proof/sumcheck/chunked_gpu_driver.h"

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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
xena::future<std::unique_ptr<workspace>>
chunked_gpu_driver::make_workspace(basct::cspan<s25t::element> mles,
                                   basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                                   basct::cspan<unsigned> product_terms,
                                   unsigned n) const noexcept {
  return {};
}
#pragma clang diagnostic pop

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
