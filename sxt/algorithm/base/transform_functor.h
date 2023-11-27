#pragma once

#include <concepts>
#include <utility>

#include "sxt/execution/async/future.h"

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// transform_functor
//--------------------------------------------------------------------------------------------------
template <class F, class ArgFirst, class... ArgsRest>
concept transform_functor = std::copy_constructible<F> &&
                        // clang-format off
  requires(const F f, ArgFirst& x1, ArgsRest&... xrest) {
    { f(x1, xrest...) } noexcept;
  };
// clang-format on

//--------------------------------------------------------------------------------------------------
// transform_functor_factory
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T, class ArgFirst, class... ArgsRest> struct match_transform_functor_future {
  static constexpr bool value = false;
};

template <class F, class ArgFirst, class... ArgsRest>
  requires transform_functor<F, ArgFirst, ArgsRest...>
struct match_transform_functor_future<xena::future<F>, ArgFirst, ArgsRest...> {
  static constexpr bool value = true;
};
} // namespace detail

template <class F, class ArgFirst, class... ArgsRest>
concept transform_functor_factory = 
                        // clang-format off
  requires(const F f) {
    requires detail::match_transform_functor_future<decltype(f()), ArgFirst, ArgsRest...>::value;
  };
// clang-format on
} // namespace sxt::algb
