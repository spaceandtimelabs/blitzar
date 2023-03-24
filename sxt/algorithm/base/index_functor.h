#pragma once

#include <concepts>

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// index_functor
//--------------------------------------------------------------------------------------------------
template <class F>
concept index_functor = std::copy_constructible<F> &&
                        // clang-format off
  requires(const F f, unsigned n, unsigned i) {
    { f(n, i) } noexcept;
  };
// clang-format on
} // namespace sxt::algb
