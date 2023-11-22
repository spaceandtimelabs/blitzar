#pragma once

#include "sxt/execution/async/future.h"

namespace sxt::algi {
//--------------------------------------------------------------------------------------------------
// transform
//--------------------------------------------------------------------------------------------------
#if 0
template <class F, class Arg1, class... ArgsRest>
  requires algb::transform_functor<F, 
           bast::value_type<Arg1>, bast::value_type<ArgsRest>...>
xena::future<> transform(
    basct::span<bast::value_type<Args1>> res,
    F f,
    basit::chunk_options chunk_options,
    const Arg1& x1,
    const ArgsRest&... xrest) noexcept {
}
#endif
} // namespace sxt::algi
