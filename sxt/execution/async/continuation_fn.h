#pragma once

#include <concepts>
#include <utility>

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// is_continuation_fn_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class F, class From, class T> struct is_continuation_fn_impl {
  static constexpr bool value = false;
};

template <class F, class From, class To>
  requires requires(F f, From from) {
    { f(std::move(from)) } noexcept -> std::convertible_to<To>;
  }
struct is_continuation_fn_impl<F, From, To> {
  static constexpr bool value = true;
};

template <class F, class To>
  requires requires(F f) {
    { f() } noexcept -> std::convertible_to<To>;
  }
struct is_continuation_fn_impl<F, void, To> {
  static constexpr bool value = true;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// continuation_fn
//--------------------------------------------------------------------------------------------------
template <class F, class From, class To>
concept continuation_fn = detail::is_continuation_fn_impl<F, From, To>::value;
} // namespace sxt::xena
