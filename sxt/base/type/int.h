#pragma once

#include <cstddef>
#include <cstdint>

using uint128_t = __uint128_t;

namespace sxt::bast {
//--------------------------------------------------------------------------------------------------
// sized_int_t_impl
//--------------------------------------------------------------------------------------------------
namespace detail {
template <size_t> struct sized_int_t_impl {};

template <> struct sized_int_t_impl<8> {
  using type = int8_t;
};

template <> struct sized_int_t_impl<16> {
  using type = int16_t;
};

template <> struct sized_int_t_impl<32> {
  using type = int32_t;
};

template <> struct sized_int_t_impl<64> {
  using type = int64_t;
};
} // namespace detail

//--------------------------------------------------------------------------------------------------
// sized_int_t
//--------------------------------------------------------------------------------------------------
template <size_t K> using sized_int_t = typename detail::sized_int_t_impl<K>::type;
} // namespace sxt::bast
