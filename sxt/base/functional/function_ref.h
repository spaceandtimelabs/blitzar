/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <memory>
#include <type_traits>

namespace sxt::basf {
//--------------------------------------------------------------------------------------------------
// function_ref
//--------------------------------------------------------------------------------------------------
template <class Sig> class function_ref;

/**
 * Non-owning function reference that can be used as a more performant
 * replacement for std::function when ownership sematics aren't needed.
 *
 * See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0792r0.html
 *
 * Based off of https://stackoverflow.com/a/39087660/4447365
 */
template <class R, class... Args> class function_ref<R(Args...)> {
  void* callable_ = nullptr;
  R (*invoker_)(void*, Args...) = nullptr;

  template <class F> using function_pointer = decltype(std::addressof(std::declval<F&>()));

  template <class F> void bind_to(F& f) noexcept {
    callable_ = reinterpret_cast<void*>(std::addressof(f));
    invoker_ = [](void* callable_, Args... args) -> R {
      return (*reinterpret_cast<function_pointer<F>>(callable_))(std::forward<Args>(args)...);
    };
  }

  template <class R_in, class... Args_in> void bind_to(R_in (*f)(Args_in...)) noexcept {
    using F = decltype(f);
    if (f == nullptr) {
      return bind_to(nullptr);
    }
    callable_ = reinterpret_cast<void*>(f);
    invoker_ = [](void* callable_, Args... args) -> R {
      return (F(callable_))(std::forward<Args>(args)...);
    };
  }

  void bind_to(std::nullptr_t) noexcept {
    callable_ = nullptr;
    invoker_ = nullptr;
  }

public:
  function_ref() noexcept = default;

  template <
      class F,
      std::enable_if_t<!std::is_same<function_ref, typename std::decay<F>::type>{}>* = nullptr,
      std::enable_if_t<std::is_convertible<typename std::result_of<F&(Args...)>::type, R>{}>* =
          nullptr>
  function_ref(F&& f) {
    bind_to(f); // not forward
  }

  function_ref(std::nullptr_t) {}

  function_ref(const function_ref&) noexcept = default;
  function_ref(function_ref&&) noexcept = default;

  function_ref& operator=(const function_ref&) noexcept = default;
  function_ref& operator=(function_ref&&) noexcept = default;

  R operator()(Args... args) const { return invoker_(callable_, std::forward<Args>(args)...); }

  explicit operator bool() const { return invoker_; }
};
} // namespace sxt::basf
