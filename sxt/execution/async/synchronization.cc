#include "sxt/execution/async/synchronization.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// await_stream
//--------------------------------------------------------------------------------------------------
future<> await_stream(bast::raw_stream_t stream) noexcept {
  promise<> p;
  future<> res{p};
  basdv::event event;
  basdv::record_event(event, stream);
  xens::get_scheduler().schedule(
      std::make_unique<gpu_computation_event<>>(std::move(event), std::move(p)));
  return res;
}
} // namespace sxt::xena
