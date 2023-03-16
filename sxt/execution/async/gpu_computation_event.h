#pragma once

#include "sxt/base/device/event.h"
#include "sxt/execution/async/computation_handle.h"
#include "sxt/execution/async/promise.h"
#include "sxt/execution/schedule/pollable_event.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// gpu_computation_event
//--------------------------------------------------------------------------------------------------
template <class T = void> class gpu_computation_event final : public xens::pollable_event {
public:
  gpu_computation_event(basdv::event&& event, computation_handle&& computation,
                        xena::promise<T>&& promise) noexcept
      : event_{std::move(event)}, computation_{std::move(computation)},
        promise_{std::move(promise)} {}

  gpu_computation_event(basdv::event&& event, xena::promise<T>&& promise) noexcept
      : event_{std::move(event)}, promise_{std::move(promise)} {}

  xena::promise<T>& promise() noexcept { return promise_; }

  // xens::pollable_event
  bool ready() noexcept override { return event_.query_is_ready(); }

  void invoke() noexcept override { promise_.make_ready(); }

private:
  basdv::event event_;
  computation_handle computation_;
  xena::promise<T> promise_;
};

extern template class gpu_computation_event<void>;
} // namespace sxt::xena
