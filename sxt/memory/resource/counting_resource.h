#pragma once

#include <memory_resource>

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// counting_resource
//--------------------------------------------------------------------------------------------------
class counting_resource final : public std::pmr::memory_resource {
public:
  counting_resource() noexcept;

  explicit counting_resource(std::pmr::memory_resource* upstream) noexcept;

  size_t bytes_allocated() const noexcept { return bytes_allocated_; }

  size_t bytes_deallocated() const noexcept { return bytes_deallocated_; }

private:
  std::pmr::memory_resource* upstream_;
  size_t bytes_allocated_ = 0;
  size_t bytes_deallocated_ = 0;

  void* do_allocate(size_t bytes, size_t alignment) noexcept override;

  void do_deallocate(void* ptr, size_t bytes, size_t alignment) noexcept override;

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;
};
} // namespace sxt::memr