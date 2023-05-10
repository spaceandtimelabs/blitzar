/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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

namespace sxt::xena {
class future_base;

//--------------------------------------------------------------------------------------------------
// promise_base
//--------------------------------------------------------------------------------------------------
class promise_base {
public:
  promise_base() noexcept = default;
  promise_base(const promise_base&) = delete;
  promise_base(promise_base&& other) noexcept;

  virtual ~promise_base() noexcept = default;

  promise_base& operator=(const promise_base&) = delete;
  promise_base& operator=(promise_base&& other) noexcept;

  void set_future(future_base* fut) noexcept { future_ = fut; }

  future_base* future() const noexcept { return future_; }

private:
  future_base* future_{nullptr};
};

//--------------------------------------------------------------------------------------------------
// future_base
//--------------------------------------------------------------------------------------------------
class future_base {
public:
  future_base() noexcept = default;
  future_base(const future_base&) = delete;
  future_base(future_base&& other) noexcept;

  virtual ~future_base() noexcept = default;

  future_base& operator=(const future_base&) = delete;
  future_base& operator=(future_base&& other) noexcept;

  void set_promise(promise_base* p) noexcept { promise_ = p; }

  promise_base* promise() const noexcept { return promise_; }

private:
  promise_base* promise_{nullptr};
};
} // namespace sxt::xena
