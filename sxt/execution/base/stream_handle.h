#pragma once

struct CUstream_st;

namespace sxt::xenb {
//--------------------------------------------------------------------------------------------------
// stream_handle
//--------------------------------------------------------------------------------------------------
struct stream_handle {
  CUstream_st* stream;
  stream_handle* next;
};
} // namespace sxt::xenb
