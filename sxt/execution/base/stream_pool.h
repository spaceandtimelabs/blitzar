#pragma once

#include <cstddef>

namespace sxt::xenb {
struct stream_handle;

//--------------------------------------------------------------------------------------------------
// stream_pool
//--------------------------------------------------------------------------------------------------
/**
 * Pool of CUDA streams.
 *
 * This allows us to cheaply aquire streams without having to continually pay the cost
 * of recreation. See https://stackoverflow.com/a/52934292 for reasons of why this is worthwhile.
 *
 * Similar to https://seastar.io/ this assumes that the application is sharded and access
 * to a particular pool will only happen on a single thread so that there is no need for
 * synchronization.
 */
class stream_pool {
public:
  explicit stream_pool(size_t initial_size) noexcept;

  ~stream_pool() noexcept;

  stream_pool(const stream_pool&) = delete;
  stream_pool(stream_pool&&) = delete;
  stream_pool& operator=(const stream_pool&) = delete;

  stream_handle* aquire_handle() noexcept;

  void release_handle(stream_handle* handle) noexcept;

private:
  stream_handle* head_;
};

//--------------------------------------------------------------------------------------------------
// get_stream_pool
//--------------------------------------------------------------------------------------------------
/**
 * Access the thread_local stream pool.
 */
stream_pool* get_stream_pool() noexcept;
} // namespace sxt::xenb
