#pragma once

#include "sxt/execution/async/shared_future.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// chunk_context
//--------------------------------------------------------------------------------------------------
struct chunk_context {
  unsigned chunk_index = 0;
  unsigned device_index = 0;
  unsigned num_devices_used = 0;
  xena::shared_future<> alt_future;
};
} // namespace sxt::xendv
