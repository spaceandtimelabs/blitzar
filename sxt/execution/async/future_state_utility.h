#pragma once

#include <type_traits>

#include "sxt/base/error/assert.h"
#include "sxt/execution/async/continuation_fn.h"
#include "sxt/execution/async/future_state.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// invoke_continuation_fn
//--------------------------------------------------------------------------------------------------
template <class T, class Tp, continuation_fn<T, Tp> F>
void invoke_continuation_fn(future_state<Tp>& state_p, F& f, future_state<T>& state) noexcept {
  SXT_DEBUG_ASSERT(state.ready());
  if constexpr (std::is_void_v<Tp>) {
    f(std::move(state.value()));
  } else {
    state_p.emplace(f(std::move(state.value())));
  }
}

template <class Tp, continuation_fn<void, Tp> F>
void invoke_continuation_fn(future_state<Tp>& state_p, F& f, future_state<void>& state) noexcept {
  SXT_DEBUG_ASSERT(state.ready());
  if constexpr (std::is_void_v<Tp>) {
    f();
  } else {
    state_p.emplace(f());
  }
}
} // namespace sxt::xena
