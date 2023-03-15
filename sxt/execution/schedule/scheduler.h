#pragma once

#include <memory>

namespace sxt::xens {
class pollable_event;

//--------------------------------------------------------------------------------------------------
// scheduler
//--------------------------------------------------------------------------------------------------
class scheduler {
public:
  void run() noexcept;

  void schedule(std::unique_ptr<pollable_event>&& event) noexcept;

private:
  std::unique_ptr<pollable_event> head_;
};

//--------------------------------------------------------------------------------------------------
// get_scheduler
//--------------------------------------------------------------------------------------------------
scheduler& get_scheduler() noexcept;
} // namespace sxt::xens
