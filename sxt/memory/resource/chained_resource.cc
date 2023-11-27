#include "sxt/memory/resource/chained_resource.h"

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
chained_resource::chained_resource() noexcept
    : chained_resource{std::pmr::get_default_resource()} {}

chained_resource::chained_resource(std::pmr::memory_resource* upstream) noexcept
    : upstream_{upstream} {}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
chained_resource::~chained_resource() noexcept {
  for (auto [ptr, bytes, alignment] : allocations_) {
    upstream_->deallocate(ptr, bytes, alignment);
  }
}

//--------------------------------------------------------------------------------------------------
// do_allocate
//--------------------------------------------------------------------------------------------------
void* chained_resource::do_allocate(size_t bytes, size_t alignment) noexcept {
  auto ptr = upstream_->allocate(bytes, alignment);
  allocations_.emplace_back(ptr, bytes, alignment);
  return ptr;
}

//--------------------------------------------------------------------------------------------------
// do_is_equal
//--------------------------------------------------------------------------------------------------
bool chained_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return this == &other;
}
} // namespace sxt::memr
