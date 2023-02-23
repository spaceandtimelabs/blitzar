#pragma once

namespace sxt::basf {
//--------------------------------------------------------------------------------------------------
// move_only_function_handle
//--------------------------------------------------------------------------------------------------
template <class T, bool> class move_only_function_handle;

template <class R, class... Args, bool IsNoexcept>
class move_only_function_handle<R(Args...), IsNoexcept> {
public:
  virtual ~move_only_function_handle() noexcept = default;

  virtual R invoke(Args...) noexcept(IsNoexcept) = 0;
};
} // namespace sxt::basf
