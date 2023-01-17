#pragma once

struct CUstream_st;

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

  CUstream_st* raw_stream() const noexcept;

private:
  stream_handle* handle_;
};
} // namespace sxt::xenb
