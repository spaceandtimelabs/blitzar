#pragma once

#include "sxt/base/type/raw_cuda_event.h"
#include "sxt/base/type/raw_stream.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// record_event
//--------------------------------------------------------------------------------------------------
void record_event(bast::raw_cuda_event_t event, bast::raw_stream_t stream) noexcept;

//--------------------------------------------------------------------------------------------------
// async_wait_on_event
//--------------------------------------------------------------------------------------------------
void async_wait_on_event(bast::raw_stream_t stream, bast::raw_cuda_event_t event) noexcept;
} // namespace sxt::basdv
