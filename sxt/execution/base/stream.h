#pragma once

#include "sxt/base/type/raw_stream.h"

namespace sxt::xenb {
struct stream_handle;

//--------------------------------------------------------------------------------------------------
// stream
//--------------------------------------------------------------------------------------------------
/**
 * Wrapper around a pooled CUDA stream.
 */
class stream {
public:
  stream() noexcept;
  stream(stream&& other) noexcept;

  ~stream() noexcept;

  stream(const stream&) = delete;
  stream& operator=(const stream&) = delete;
  stream& operator=(stream&& other) noexcept;

  stream_handle* release_handle() noexcept;

  bast::raw_stream_t raw_stream() const noexcept;

  operator bast::raw_stream_t() const noexcept { return this->raw_stream(); }

private:
  stream_handle* handle_;
};
} // namespace sxt::xenb
