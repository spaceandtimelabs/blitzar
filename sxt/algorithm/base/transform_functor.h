#pragma once

#include <concepts>

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// transform_functor
//--------------------------------------------------------------------------------------------------
template <class F, class ArgFirst, class ArgSecond, class... ArgsRest>
concept transform_functor = std::copy_constructible<F> &&
                        // clang-format off
  requires(const F f, ArgFirst& x1, const ArgSecond& x2, const ArgsRest&... xrest) {
    { f(x1, x2, xrest...) } noexcept;
  };
// clang-format on
} // namespace sxt::algb

