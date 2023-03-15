#pragma once

#include <memory>

namespace sxt::xens {
//--------------------------------------------------------------------------------------------------
// pollable_event
//--------------------------------------------------------------------------------------------------
class pollable_event {
public:
  virtual ~pollable_event() noexcept = default;

  void set_next(std::unique_ptr<pollable_event>&& next) noexcept;

  std::unique_ptr<pollable_event> release_next() noexcept { return {std::move(next_)}; }

  pollable_event* next() noexcept { return next_.get(); }

  virtual bool ready() noexcept = 0;

  virtual void invoke() noexcept = 0;

private:
  std::unique_ptr<pollable_event> next_;
};
} // namespace sxt::xens
