#pragma once

#include <cstddef>
#include <memory_resource>

namespace sxt::bast {
//--------------------------------------------------------------------------------------------------
// polymorphic_allocator
//--------------------------------------------------------------------------------------------------
using polymorphic_allocator = std::pmr::polymorphic_allocator<std::byte>;
} // namespace sxt::bast
