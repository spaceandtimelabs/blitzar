#include "sxt/memory/resource/counting_resource.h"

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
counting_resource::counting_resource() noexcept
    : counting_resource{std::pmr::get_default_resource()} {}

counting_resource::counting_resource(std::pmr::memory_resource* upstream) noexcept
    : upstream_{upstream} {}

//--------------------------------------------------------------------------------------------------
// do_allocate
//--------------------------------------------------------------------------------------------------
void* counting_resource::do_allocate(size_t bytes, size_t alignment) noexcept {
  bytes_allocated_ += bytes;
  return upstream_->allocate(bytes, alignment);
}

//--------------------------------------------------------------------------------------------------
// do_deallocate
//--------------------------------------------------------------------------------------------------
void counting_resource::do_deallocate(void* ptr, size_t bytes, size_t alignment) noexcept {
  bytes_deallocated_ += bytes;
  upstream_->deallocate(ptr, bytes, alignment);
}

//--------------------------------------------------------------------------------------------------
// do_is_equal
//--------------------------------------------------------------------------------------------------
bool counting_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return this == &other;
}
} // namespace sxt::memr

