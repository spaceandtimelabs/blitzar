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
