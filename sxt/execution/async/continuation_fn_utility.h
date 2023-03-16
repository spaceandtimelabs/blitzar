#pragma once

#include <type_traits>

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// continuation_fn_result_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class F, class T> struct continuation_fn_result_impl {
  using type = std::invoke_result_t<F&, T&&>;
};

template <class F> struct continuation_fn_result_impl<F, void> {
  using type = std::invoke_result_t<F&>;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// continuation_fn_result_t
//--------------------------------------------------------------------------------------------------
template <class F, class T>
using continuation_fn_result_t = typename detail::continuation_fn_result_impl<F, T>::type;
} // namespace sxt::xena
