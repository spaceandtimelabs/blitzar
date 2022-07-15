#pragma once

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// stream
//--------------------------------------------------------------------------------------------------
class stream {
public:
  // constructor
  stream() noexcept;

  ~stream() noexcept;

  /* Prohibits from receiving another stream */
  stream(const stream& other) = delete;

  stream(stream&& other) = delete;

  stream& operator=(stream&&) = delete;

  stream& operator=(const stream&) = delete;

  cudaStream_t raw_stream() noexcept { return stream_; }

private:
  cudaStream_t stream_;
};

} // namespace sxt::basdv
