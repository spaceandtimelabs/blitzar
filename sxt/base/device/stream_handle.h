#pragma once

#include "sxt/base/type/raw_stream.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// stream_handle
//--------------------------------------------------------------------------------------------------
struct stream_handle {
  bast::raw_stream_t stream;
  stream_handle* next;
};
} // namespace sxt::basdv
