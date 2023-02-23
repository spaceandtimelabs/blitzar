#pragma once

#include <concepts>
#include <functional>
#include <type_traits>

#include "sxt/base/functional/move_only_function_handle.h"

namespace sxt::basf {
//--------------------------------------------------------------------------------------------------
// move_only_function_handle_impl
//--------------------------------------------------------------------------------------------------
template <class, class, bool> class move_only_function_handle_impl;

template <class F, class R, class... Args, bool IsNoexcept>
  requires std::is_invocable_r_v<R, F, Args...>
class move_only_function_handle_impl<F, R(Args...), IsNoexcept> final
    : public move_only_function_handle<R(Args...), IsNoexcept> {
public:
  explicit move_only_function_handle_impl(F&& f) noexcept : f_{std::move(f)} {}

  R invoke(Args... args) noexcept(IsNoexcept) override {
    return std::invoke(f_, std::forward<Args>(args)...);
  }

private:
  F f_;
};
} // namespace sxt::basf
