#pragma once

namespace sxt::bascpt {
//--------------------------------------------------------------------------------------------------
// field
//--------------------------------------------------------------------------------------------------
template <class T>
concept field = requires(T& res, const T& e) {
  neg(res, e);
  add(res, e, e);
  sub(res, e, e);
  mul(res, e, e);
  muladd(res, e, e, e);
};
} // namespace sxt::bascpt
