#include "sxt/proof/sumcheck/chunked_gpu_driver.h"

#include <algorithm>
#include <print>

#include "sxt/algorithm/iteration/transform.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/proof/sumcheck/device_cache.h"
#include "sxt/proof/sumcheck/fold_gpu.h"
#include "sxt/proof/sumcheck/gpu_driver.h"
#include "sxt/proof/sumcheck/mle_utility.h"
#include "sxt/proof/sumcheck/sum_gpu.h"
#include "sxt/scalar25/operation/sub.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// chunked_gpu_workspace
//--------------------------------------------------------------------------------------------------
namespace {
struct chunked_gpu_workspace final : public workspace {
  std::unique_ptr<workspace> single_gpu_workspace;

  device_cache cache;
  memmg::managed_array<s25t::element> mles_data;
  basct::cspan<s25t::element> mles;
  unsigned n;
  unsigned num_variables;

  chunked_gpu_workspace(basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                        basct::cspan<unsigned> product_terms) noexcept
      : cache{product_table, product_terms} {}
};
} // namespace

//--------------------------------------------------------------------------------------------------
// try_make_single_gpu_workspace 
//--------------------------------------------------------------------------------------------------
static xena::future<> try_make_single_gpu_workspace(chunked_gpu_workspace& work,
                                                    double no_chunk_cutoff) noexcept {
  auto gpu_memory_fraction = get_gpu_memory_fraction(work.mles);
  if (gpu_memory_fraction > no_chunk_cutoff) {
    co_return;
  }

  // construct single gpu workspace
  auto cache_data = work.cache.clear();
  gpu_driver drv;
  work.single_gpu_workspace =
      co_await drv.make_workspace(work.mles, std::move(cache_data->product_table),
                                  std::move(cache_data->product_terms), work.n);

  // free data we no longer need
  work.mles_data.reset();
  work.mles = {};
}

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
chunked_gpu_driver::chunked_gpu_driver(double no_chunk_cutoff) noexcept
    : no_chunk_cutoff_{no_chunk_cutoff} {
  SXT_RELEASE_ASSERT(0 <= no_chunk_cutoff_ && no_chunk_cutoff_ <= 1.0);
}

//--------------------------------------------------------------------------------------------------
// make_workspace
//--------------------------------------------------------------------------------------------------
xena::future<std::unique_ptr<workspace>>
chunked_gpu_driver::make_workspace(basct::cspan<s25t::element> mles,
                                   basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                                   basct::cspan<unsigned> product_terms,
                                   unsigned n) const noexcept {
  auto res = std::make_unique<chunked_gpu_workspace>(product_table, product_terms);
  res->mles = mles;
  res->n = n;
  res->num_variables = std::max(basn::ceil_log2(n), 1);
  auto gpu_memory_fraction = get_gpu_memory_fraction(mles);
  if (gpu_memory_fraction <= no_chunk_cutoff_) {
    gpu_driver drv;
    res->single_gpu_workspace = co_await drv.make_workspace(mles, product_table, product_terms, n);
  }
  co_return std::unique_ptr<workspace>(std::move(res));
}

//--------------------------------------------------------------------------------------------------
// sum
//--------------------------------------------------------------------------------------------------
xena::future<> chunked_gpu_driver::sum(basct::span<s25t::element> polynomial,
                                       workspace& ws) const noexcept {
  std::println(stderr, "*********************");
  auto& work = static_cast<chunked_gpu_workspace&>(ws);
  if (work.single_gpu_workspace) {
    gpu_driver drv;
    co_return co_await drv.sum(polynomial, *work.single_gpu_workspace);
  }
  co_await sum_gpu(polynomial, work.cache, work.mles, work.n);
}

//--------------------------------------------------------------------------------------------------
// fold
//--------------------------------------------------------------------------------------------------
xena::future<> chunked_gpu_driver::fold(workspace& ws, const s25t::element& r) const noexcept {
  using s25t::operator""_s25;
  auto& work = static_cast<chunked_gpu_workspace&>(ws);
  if (work.single_gpu_workspace) {
    gpu_driver drv;
    co_return co_await drv.fold(*work.single_gpu_workspace, r);
  }
  auto n = work.n;
  auto mid = 1u << (work.num_variables - 1u);
  auto num_mles = work.mles.size() / n;
  SXT_RELEASE_ASSERT(
      // clang-format off
      work.n >= mid && work.mles.size() % n == 0
      // clang-format on
  );

  s25t::element one_m_r = 0x1_s25;
  s25o::sub(one_m_r, one_m_r, r);

  // fold
  memmg::managed_array<s25t::element> mles_p(num_mles * mid);
  co_await fold_gpu(mles_p, work.mles, n, r);

  // update
  work.n = mid;
  --work.num_variables;
  work.mles_data = std::move(mles_p);
  work.mles = work.mles_data;

  // check if we should fall back to single gpu workspace
  co_await try_make_single_gpu_workspace(work, no_chunk_cutoff_);
}
} // namespace sxt::prfsk
