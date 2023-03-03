#pragma once

#include <memory_resource>

#include "sxt/base/type/raw_stream.h"

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// async_device_resource
//--------------------------------------------------------------------------------------------------
class async_device_resource final : public std::pmr::memory_resource {
public:
  explicit async_device_resource(bast::raw_stream_t stream) noexcept;

private:
  bast::raw_stream_t stream_;

  void* do_allocate(size_t bytes, size_t alignment) noexcept override;

  void do_deallocate(void* ptr, size_t bytes, size_t alignment) noexcept override;

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;
};
} // namespace sxt::memr
