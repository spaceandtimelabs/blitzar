#pragma once

#include "sxt/algorithm/base/transform_functor.h"
#include "sxt/base/container/span.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/chunk_options.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/base/type/value_type.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/for_each.h"

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
      auto f_fut = make_f();

      auto f = co_await std::move(f_fut);
      (void)f;
  });
}
} // namespace sxt::algi
