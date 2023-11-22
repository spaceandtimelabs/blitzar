#pragma once

namespace sxt::bast {
//--------------------------------------------------------------------------------------------------
// value_type
//--------------------------------------------------------------------------------------------------
template <class T>
struct value_type {};

template <class T>
  requires requires {
    typename T::value_type;
  }
struct value_type<T> {
  using type = typename T::value_type;
};

//--------------------------------------------------------------------------------------------------
// value_type_t
//--------------------------------------------------------------------------------------------------
template <class T>
using value_type_t = typename value_type<T>::type;
} // namespace sxt::bast
