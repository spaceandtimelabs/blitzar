#pragma once

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
};
} // namespace sxt::basfld
