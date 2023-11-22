#pragma once

#include "sxt/algorithm/base/transform_functor.h"
#include "sxt/base/container/span.h"
#include "sxt/base/iterator/chunk_options.h"
#include "sxt/base/type/value_type.h"
#include "sxt/execution/async/future.h"

namespace sxt::algi {
//--------------------------------------------------------------------------------------------------
// transform
//--------------------------------------------------------------------------------------------------
template <class F, class Arg1, class... ArgsRest>
  requires algb::transform_functor<F, bast::value_type_t<Arg1>, bast::value_type_t<ArgsRest>...>
xena::future<> transform(basct::span<bast::value_type<Arg1>> res, F f,
                         basit::chunk_options chunk_options, const Arg1& x1,
                         const ArgsRest&... xrest) noexcept {
  return {};
}
} // namespace sxt::algi
