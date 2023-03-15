#pragma once

#include <functional>

#include "sxt/execution/schedule/pollable_event.h"

namespace sxt::xens {
//--------------------------------------------------------------------------------------------------
// test_pollable_event
//--------------------------------------------------------------------------------------------------
class test_pollable_event final : public pollable_event {
public:
  test_pollable_event(int id, int counter, std::function<void(int)> f = {}) noexcept;

  // pollable_event
  bool ready() noexcept override;

  void invoke() noexcept override;

private:
  int id_;
  mutable int counter_;
  std::function<void(int)> f_;
};
} // namespace sxt::xens
