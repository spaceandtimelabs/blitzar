#pragma once

#include <concepts>

namespace sxt::basfld {
//--------------------------------------------------------------------------------------------------
// element
//--------------------------------------------------------------------------------------------------
template <class T>
concept element = requires(T& res, const T& e) {
  neg(res, e);
  add(res, e, e);
  sub(res, e, e);
  mul(res, e, e);
  muladd(res, e, e, e);
  { T::identity() } noexcept -> std::same_as<T>;
  { T::one() } noexcept -> std::same_as<T>;
};
} // namespace sxt::basfld
