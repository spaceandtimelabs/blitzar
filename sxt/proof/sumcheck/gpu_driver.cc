#include "sxt/proof/sumcheck/gpu_driver.h"

#include <algorithm>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/algorithm/reduction/reduction.h"
#include "sxt/base/container/stack_array.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/panic.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/constexpr_switch.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/proof/sumcheck/polynomial_mapper.h"
#include "sxt/proof/sumcheck/polynomial_reducer.h"
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
  /* basdv::stream stream; */
  /* memr::async_device_resource resource; */
  memmg::managed_array<s25t::element> mles;
  memmg::managed_array<std::pair<s25t::element, unsigned>> product_table;
  memmg::managed_array<unsigned> product_terms;
  unsigned n;
  unsigned num_variables;

#if 0
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
#endif

  gpu_workspace() noexcept
      : mles{memr::get_device_resource()}, product_table{memr::get_device_resource()},
        product_terms{memr::get_device_resource()} {}
};
} // namespace

//--------------------------------------------------------------------------------------------------
// make_workspace
//--------------------------------------------------------------------------------------------------
xena::future<std::unique_ptr<workspace>>
gpu_driver::make_workspace(basct::cspan<s25t::element> mles,
                           basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                           basct::cspan<unsigned> product_terms, unsigned n) const noexcept {
  auto ws = std::make_unique<gpu_workspace>();

  // dimensions
  ws->n = n;
  ws->num_variables = basn::ceil_log2(n);

  // mles
  ws->mles = memmg::managed_array<s25t::element>{
      mles.size(),
      memr::get_device_resource(),
  };
  basdv::stream mle_stream;
  basdv::async_copy_host_to_device(ws->mles, mles, mle_stream);

  // product_table
  ws->product_table = memmg::managed_array<std::pair<s25t::element, unsigned>>{
      product_table.size(),
      memr::get_device_resource(),
  };
  basdv::stream product_table_stream;
  basdv::async_copy_host_to_device(ws->product_table, product_table, product_table_stream);

  // product_terms
  ws->product_terms = memmg::managed_array<unsigned>{
      product_terms.size(),
      memr::get_device_resource(),
  };
  basdv::stream product_terms_stream;
  basdv::async_copy_host_to_device(ws->product_terms, product_terms, product_terms_stream);

  // await
  co_await xendv::await_stream(mle_stream);
  co_await xendv::await_stream(product_table_stream);
  co_await xendv::await_stream(product_terms_stream);
  co_return ws;
}

//--------------------------------------------------------------------------------------------------
// sum
//--------------------------------------------------------------------------------------------------
xena::future<> gpu_driver::sum(basct::span<s25t::element> polynomial,
                             workspace& ws) const noexcept {
  static constexpr unsigned max_degree_v = 5u;
  auto& work = static_cast<gpu_workspace&>(ws);
  auto n = work.n;
  auto mid = 1u << (work.num_variables - 1u);
  SXT_RELEASE_ASSERT(
      // clang-format off
      work.n > mid &&
      polynomial.size() - 1u <= max_degree_v
      // clang-format on
  );

  xena::future<> res;
  auto f = [&]<unsigned MaxDegree>(std::integral_constant<unsigned, MaxDegree>) noexcept {
    polynomial_mapper<MaxDegree> mapper{
        .mles = work.mles.data(),
        .product_table = work.product_table.data(),
        .product_terms = work.product_terms.data(),
        .num_products = static_cast<unsigned>(work.product_terms.size()),
        .mid = mid,
        .n = n,
    };
    auto fut = algr::reduce<polynomial_reducer<MaxDegree>>(basdv::stream{}, mapper, mid);
    res = fut.then([&](std::array<s25t::element, MaxDegree + 1u> p) noexcept {
      for (unsigned i = 0; i < p.size(); ++i) {
        polynomial[i] = p[i];
      }
    });
  };
  basn::constexpr_switch<1u, max_degree_v + 1u>(polynomial.size() - 1u, f);
  co_await std::move(res);
}

//--------------------------------------------------------------------------------------------------
// fold
//--------------------------------------------------------------------------------------------------
xena::future<> gpu_driver::fold(workspace& ws, const s25t::element& r) const noexcept {
  (void)ws;
  (void)r;
  return xena::make_ready_future();
#if 0
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
#endif
}
} // namespace prfsk

