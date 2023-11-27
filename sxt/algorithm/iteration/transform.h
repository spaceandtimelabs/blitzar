#pragma once

#include <tuple>

#include "sxt/algorithm/base/transform_functor.h"
#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/chunk_options.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/base/type/value_type.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/device_viewable.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/chained_resource.h"

namespace sxt::algi {
//--------------------------------------------------------------------------------------------------
// transform
//--------------------------------------------------------------------------------------------------
template <class F, class Arg1, class... ArgsRest>
  requires algb::transform_functor_factory<F, bast::value_type_t<Arg1>,
                                           bast::value_type_t<ArgsRest>...>
xena::future<> transform(basct::span<bast::value_type<Arg1>> res, F make_f,
                         basit::chunk_options chunk_options, const Arg1& x1,
                         const ArgsRest&... xrest) noexcept {
  auto n = res.size();
  SXT_DEBUG_ASSERT(
      x1.size() == n && ((xrest.size() == n) && ...)
  );
  auto [first, last] = basit::split(basit::index_range{0, n}
                                        .min_chunk_size(chunk_options.min_size)
                                        .max_chunk_size(chunk_options.max_size),
                                    chunk_options.split_factor);
  co_await xendv::concurrent_for_each(
      first, last, [&](const basit::index_range& rng) noexcept -> xena::future<> { 
      basdv::stream stream;
      memr::async_device_resource resource{stream};
      memr::chained_resource alloc{&resource};

      auto x1_slice_fut =
          xendv::make_active_device_viewable(&alloc, basct::subspan(x1, rng.a(), rng.size()));
      auto xrest_slices_fut = std::make_tuple(xendv::make_active_device_viewable(
          &alloc, basct::subspan(xrest, rng.a(), rng.size()))...);

      auto f = co_await make_f(&alloc, stream);
      auto x1_slice = co_await std::move(x1_slice_fut);
      (void)x1_slice;
      (void)xrest_slices_fut;
      (void)f;
  });
}
} // namespace sxt::algi
