#include "sxt/execution/base/stream.h"

#include <cassert>

#include "sxt/execution/base/stream_handle.h"
#include "sxt/execution/base/stream_pool.h"

namespace sxt::xenb {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
stream::stream() noexcept { handle_ = get_stream_pool()->aquire_handle(); }

stream::stream(stream&& other) noexcept { handle_ = other.release_handle(); }

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
stream::~stream() noexcept {
  if (handle_ == nullptr) {
    return;
  }
  get_stream_pool()->release_handle(handle_);
}

//--------------------------------------------------------------------------------------------------
// operator=
//--------------------------------------------------------------------------------------------------
stream& stream::operator=(stream&& other) noexcept {
  if (handle_ != nullptr) {
    get_stream_pool()->release_handle(handle_);
  }
  handle_ = other.release_handle();
  return *this;
}

//--------------------------------------------------------------------------------------------------
// release_handle
//--------------------------------------------------------------------------------------------------
stream_handle* stream::release_handle() noexcept {
  auto res = handle_;
  handle_ = nullptr;
  return res;
}

//--------------------------------------------------------------------------------------------------
// raw_stream
//--------------------------------------------------------------------------------------------------
CUstream_st* stream::raw_stream() const noexcept {
  assert(handle_ != nullptr);
  return handle_->stream;
}
} // namespace sxt::xenb
