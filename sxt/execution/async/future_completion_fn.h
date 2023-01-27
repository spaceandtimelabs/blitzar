#pragma once

#include <functional>

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// future_completion_fn_type
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T> struct future_completion_fn_type {
  using type = std::function<void(T&)>;
};

template <> struct future_completion_fn_type<void> {
  using type = std::function<void()>;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// future_completion_fn
//--------------------------------------------------------------------------------------------------
template <class T> using future_completion_fn = typename detail::future_completion_fn_type<T>::type;
} // namespace sxt::xena
