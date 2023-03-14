#pragma once

#include "sxt/base/type/raw_cuda_event.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// event
//--------------------------------------------------------------------------------------------------
class event {
public:
  event() noexcept;
  event(const event&) = delete;
  event(event&& other) noexcept;

  ~event() noexcept;

  event& operator=(const event&) = delete;
  event& operator=(event&& other) noexcept;

  operator const CUevent_st*() const noexcept { return event_; }

  operator bast::raw_cuda_event_t() noexcept { return event_; }

  void clear() noexcept;

  bool query_is_ready() noexcept;

private:
  bast::raw_cuda_event_t event_;
};
} // namespace sxt::basdv
