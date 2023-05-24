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

#include <type_traits>
#include <utility>

#include "sxt/execution/async/continuation.h"
#include "sxt/execution/async/continuation_fn.h"
#include "sxt/execution/async/continuation_fn_utility.h"
#include "sxt/execution/async/future_fwd.h"
#include "sxt/execution/async/future_state.h"
#include "sxt/execution/async/future_state_utility.h"
#include "sxt/execution/async/leaf_continuation.h"
#include "sxt/execution/async/promise.h"
#include "sxt/execution/async/promise_future_base.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// future
//--------------------------------------------------------------------------------------------------
/**
 * Manage asynchronous computations.
 *
 * This is a highly simplified version of a future derived from seastar
 *
 * See https://seastar.io/futures-promises/
 */
template <class T> class future final : protected future_base {
public:
  using value_type = T;
  using reference = std::add_lvalue_reference_t<T>;

  future() noexcept = default;
  future(const future&) = delete;
  future(future&& other) noexcept
      : future_base{std::move(static_cast<future_base&>(other))}, state_{std::move(other.state_)} {
    if (auto ps = this->promise(); ps != nullptr) {
      ps->set_state(state_);
    }
  }
  explicit future(future_state<T>&& state) noexcept : state_{std::move(state)} {}
  explicit future(xena::promise<T>& p) noexcept {
    this->set_promise(&p);
    p.set_future(this);
    p.set_state(state_);
  }
  future(xena::promise<T>& p, future_state<T>&& state) noexcept : future{p} {
    state_ = std::move(state);
  }

  ~future() noexcept { this->try_run_empty_continuation(); }

  future& operator=(const future&) = delete;
  future& operator=(future&& other) noexcept {
    this->try_run_empty_continuation();
    *static_cast<future_base*>(this) = std::move(static_cast<future_base&>(other));
    state_ = std::move(other.state_);
    if (auto ps = this->promise(); ps != nullptr) {
      ps->set_state(state_);
    }
    return *this;
  }

  bool ready() const noexcept { return state_.ready(); }

  reference value() & noexcept
    requires(!std::is_void_v<T>)
  {
    return state_.value();
  }

  T value() && noexcept
    requires(!std::is_void_v<T>)
  {
    return T{std::move(state_.value())};
  }

  const reference value() const& noexcept
    requires(!std::is_void_v<T>)
  {
    return *state_.value();
  }

  xena::promise<T>* promise() const noexcept {
    return static_cast<xena::promise<T>*>(static_cast<const future_base*>(this)->promise());
  }

  template <class F, class Tp = continuation_fn_result_t<F, T>>
  auto then(F&& f) noexcept -> future<Tp>
    requires continuation_fn<F, T, Tp>
  {
    if (state_.ready()) {
      future_state<Tp> state_p;
      invoke_continuation_fn(state_p, f, state_);
      return future<Tp>{std::move(state_p)};
    }
    auto ps = this->promise();
    SXT_DEBUG_ASSERT(ps != nullptr, "promise must be set");
    // Note: continuation will be deleted when the associated promise is made ready
    auto cont =
        new continuation<T, Tp, std::remove_reference_t<F>>{std::move(state_), std::move(f)};
    ps->set_state(cont->state());
    ps->set_continuation(*cont);
    ps->set_future(nullptr);
    this->set_promise(nullptr);
    return future<Tp>{cont->promise_p()};
  }

  const future_state<T>& state() const noexcept { return state_; }

private:
  future_state<T> state_;

  void try_run_empty_continuation() noexcept {
    if (state_.ready()) {
      return;
    }
    auto ps = this->promise();
    if (ps == nullptr) {
      return;
    }

    // run an empty continuation so that when the associated promise is ready it
    // will have a valid future_state
    auto cont = new leaf_continuation<T>{std::move(state_)};
    ps->set_state(cont->state());
    ps->set_continuation(*cont);
    ps->set_future(nullptr);
    this->set_promise(nullptr);
  }
};

extern template class future<void>;

//--------------------------------------------------------------------------------------------------
// make_ready_future
//--------------------------------------------------------------------------------------------------
template <class T> future<T> make_ready_future(T&& value) noexcept {
  future_state<T> state;
  state.emplace(std::move(value));
  state.make_ready();
  return future<T>{std::move(state)};
}

future<> make_ready_future() noexcept;
} // namespace sxt::xena
