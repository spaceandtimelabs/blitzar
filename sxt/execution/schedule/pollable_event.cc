#include "sxt/execution/schedule/pollable_event.h"

namespace sxt::xens {
//--------------------------------------------------------------------------------------------------
// set_next
//--------------------------------------------------------------------------------------------------
void pollable_event::set_next(std::unique_ptr<pollable_event>&& next) noexcept {
  next_ = std::move(next);
}
} // namespace sxt::xens
