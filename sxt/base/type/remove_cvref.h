#pragma once

#include <type_traits>

namespace sxt::bast {
//--------------------------------------------------------------------------------------------------
// remove_cvref
//--------------------------------------------------------------------------------------------------
// c++17 version of std::remove_cvref
// see https://en.cppreference.com/w/cpp/types/remove_cvref
template< class T >
struct remove_cvref {
    typedef std::remove_cv_t<std::remove_reference_t<T>> type;
};

//--------------------------------------------------------------------------------------------------
// remove_cvref_t
//--------------------------------------------------------------------------------------------------
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;
} // namespace sxt::bast
