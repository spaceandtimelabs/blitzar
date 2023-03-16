#include "sxt/execution/async/future.h"

namespace sxt::xena {
template class future<void>;

//--------------------------------------------------------------------------------------------------
// make_ready_future
//--------------------------------------------------------------------------------------------------
future<> make_ready_future() noexcept {
  future_state<void> state;
  state.make_ready();
  return future<>{std::move(state)};
}
} // namespace sxt::xena
