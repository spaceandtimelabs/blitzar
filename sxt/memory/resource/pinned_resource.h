#pragma once

#include <memory_resource>

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// pinned_resource
//--------------------------------------------------------------------------------------------------
class pinned_resource final : public std::pmr::memory_resource {
public:
private:
  void* do_allocate(size_t bytes, size_t alignment) noexcept override;

  void do_deallocate(void* ptr, size_t bytes, size_t alignment) noexcept override;

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;
};

//--------------------------------------------------------------------------------------------------
// get_pinned_resource
//--------------------------------------------------------------------------------------------------
pinned_resource* get_pinned_resource() noexcept;
} // namespace sxt::memr
