#pragma once

#include <concepts>

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
template <class F, class ArgFirst, class... ArgsRest>
concept transform_functor_factory = 
                        // clang-format off
  requires(const F f) {
    { f() } noexcept -> transform_functor<F, ArgFirst, ArgsRest...>;
  };
// clang-format on
} // namespace sxt::algb
