#pragma once

#include <cstddef>
#include <memory_resource>

namespace sxt::basm {
//--------------------------------------------------------------------------------------------------
// alloc_t
//--------------------------------------------------------------------------------------------------
using alloc_t = std::pmr::polymorphic_allocator<std::byte>;
} // namespace sxt::basm
