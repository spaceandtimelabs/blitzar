#include "sxt/proof/sumcheck/fold_gpu.h"

#include "sxt/algorithm/iteration/kernel_fit.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/scalar25/operation/sub.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// fold_impl 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
static xena::future<> fold_impl(basct::span<s25t::element> mles, unsigned n, unsigned a, unsigned b,
                                const s25t::element& r, const s25t::element one_m_r) noexcept {
/* void copy_partial_mles(memmg::managed_array<s25t::element>& partial_mles, basdv::stream& stream, */
/*                        basct::cspan<s25t::element> mles, unsigned n, unsigned a, */
/*                        unsigned b) noexcept; */
  return {};
}
#pragma clang diagnostic pop

//--------------------------------------------------------------------------------------------------
// fold_gpu 
//--------------------------------------------------------------------------------------------------
xena::future<> fold_gpu(basct::span<s25t::element> mles, unsigned n, const s25t::element& r) noexcept {
  using s25t::operator""_s25;
  auto num_mles = mles.size() / n;
  auto mid = n / 2;
  SXT_DEBUG_ASSERT(
      n > 1 && mles.size() == num_mles * n
  );
  s25t::element one_m_r = 0x1_s25;
  s25o::sub(one_m_r, one_m_r, r);

  // split
  auto [chunk_first, chunk_last] = basit::split(
      basit::index_range{0, mid}.min_chunk_size(1024 * 512).max_chunk_size(1024 * 1024 * 5),
      basdv::get_num_devices());

  // fold
  co_await xendv::concurrent_for_each(chunk_first, chunk_last,
                                      [&](basit::index_range rng) noexcept -> xena::future<> {
                                        co_await fold_impl(mles, n, rng.a(), rng.b(), r, one_m_r);
                                      });
}
} // namespace sxt::prfsk
