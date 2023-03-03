#pragma once

#include "sxt/base/type/raw_cuda_event.h"
#include "sxt/base/type/raw_stream.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// record_event
//--------------------------------------------------------------------------------------------------
void record_event(bast::raw_cuda_event_t event, bast::raw_stream_t stream) noexcept;
} // namespace sxt::basdv
