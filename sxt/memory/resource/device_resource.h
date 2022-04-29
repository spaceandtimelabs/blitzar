#pragma once

#include <memory_resource>

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// device_resource
//--------------------------------------------------------------------------------------------------
class device_resource final : public std::pmr::memory_resource {
 public:
 private:
   void* do_allocate(size_t bytes, size_t alignment) noexcept override;

   void do_deallocate(void* ptr, size_t bytes, size_t alignment) noexcept override;

   bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;
};

//--------------------------------------------------------------------------------------------------
// get_device_resource
//--------------------------------------------------------------------------------------------------
device_resource* get_device_resource() noexcept;
} // namespace sxt::memr
