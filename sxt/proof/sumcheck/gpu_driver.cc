#include "sxt/proof/sumcheck/gpu_driver.h"

#include <algorithm>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/stack_array.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/panic.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/operation/sub.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// gpu_workspace
//--------------------------------------------------------------------------------------------------
namespace {
struct gpu_workspace final : public workspace {
  basdv::stream stream;
  memr::async_device_resource resource;
  memmg::managed_array<s25t::element> mles;
  memmg::managed_array<std::pair<s25t::element, unsigned>> product_table;
  memmg::managed_array<unsigned> product_terms;
  unsigned n;
  unsigned num_variables;

  gpu_workspace(basct::cspan<s25t::element> mles_p,
                basct::cspan<std::pair<s25t::element, unsigned>> product_table_p,
                basct::cspan<unsigned> product_terms_p, unsigned np) noexcept
      : resource{stream}, mles{mles_p.size(), &resource},
        product_table{product_table_p.size(), &resource},
        product_terms{product_terms_p.size(), &resource}, n{np},
        num_variables{static_cast<unsigned>(basn::ceil_log2(np))} {
    basdv::async_copy_host_to_device(mles, mles_p, stream);
    basdv::async_copy_host_to_device(product_table, product_table_p, stream);
    basdv::async_copy_host_to_device(product_terms, product_terms_p, stream);
  }
};
} // namespace

//--------------------------------------------------------------------------------------------------
// make_workspace
//--------------------------------------------------------------------------------------------------
std::unique_ptr<workspace>
gpu_driver::make_workspace(basct::cspan<s25t::element> mles,
                           basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                           basct::cspan<unsigned> product_terms, unsigned n) const noexcept {
  return std::make_unique<gpu_workspace>(mles, product_table, product_terms, n);
}

//--------------------------------------------------------------------------------------------------
// sum
//--------------------------------------------------------------------------------------------------
xena::future<> gpu_driver::sum(basct::span<s25t::element> polynomial,
                             workspace& ws) const noexcept {
  auto& work = static_cast<gpu_workspace&>(ws);
  auto n = work.n;
  auto mid = 1u << (work.num_variables - 1u);
  SXT_RELEASE_ASSERT(work.n > mid);

  (void)polynomial;
  (void)ws;
#if 0
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
      SXT_STACK_ARRAY(p, num_terms + 1u, s25t::element);
      expand_products(p, mles + i, n, mid, terms);
      for (unsigned term_index = 0; term_index < p.size(); ++term_index) {
        s25o::muladd(polynomial[term_index], mult, p[term_index], polynomial[term_index]);
      }
      term_first += num_terms;
    }
  }
  auto n2 = mid - n1;
  SXT_RELEASE_ASSERT(n2 == 0, "not implemented yet");
#endif
  return xena::make_ready_future();
}

//--------------------------------------------------------------------------------------------------
// fold
//--------------------------------------------------------------------------------------------------
xena::future<> gpu_driver::fold(workspace& ws, const s25t::element& r) const noexcept {
  using s25t::operator""_s25;
  auto& work = static_cast<gpu_workspace&>(ws);
  auto n = work.n;
  auto mid = 1u << (work.num_variables - 1u);
  auto num_mles = work.mles.size() / n;
  SXT_RELEASE_ASSERT(
      // clang-format off
      work.n > mid && work.mles.size() % n == 0
      // clang-format on
  );

  auto n1 = work.n - mid;
  s25t::element one_m_r = 0x1_s25;
  s25o::sub(one_m_r, one_m_r, r);

  // f1
  auto f1 = [
    // clang-format off
    mles = work.mles.data(),
    n = n,
    num_mles = num_mles,
    mid = mid,
    r = r,
    one_m_r = one_m_r
    // clang-format on
  ] __device__ __host__ (unsigned /*n1*/, unsigned i) noexcept {
    for (unsigned mle_index=0; mle_index<num_mles; ++mle_index) {
      auto data = mles + n * mle_index;
      auto val = data[i];
      s25o::mul(val, val, one_m_r);
      s25o::muladd(val, r, data[mid + i], val);
      data[i] = val;
    }
  };
  algi::for_each(f1, n1);

  SXT_RELEASE_ASSERT(n1 == mid, "not implemented yet");

  work.n = mid;
  --work.num_variables;
  return xendv::await_stream(work.stream);
  (void)f1;
/* template <algb::index_functor F> __global__ void for_each_kernel(F f, unsigned n) { */
  (void)ws;
  (void)r;
#if 0
  using s25t::operator""_s25;


  auto mles = work.mles.data();
  s25t::element one_m_r = 0x1_s25;
  s25o::sub(one_m_r, one_m_r, r);
  auto n1 = work.n - mid;
  for (auto mle_index = 0; mle_index < num_mles; ++mle_index) {
    auto data = mles + n * mle_index;
    for (unsigned i = 0; i < n1; ++i) {
      auto val = data[i];
      s25o::mul(val, val, one_m_r);
      s25o::muladd(val, r, data[mid + i], val);
      data[i] = val;
    }
  }

  auto n2 = mid - n1;
  SXT_RELEASE_ASSERT(n2 == 0, "not implemented yet");
  work.n = mid;
  --work.num_variables;
#endif
  return xena::make_ready_future();
}
} // namespace prfsk

