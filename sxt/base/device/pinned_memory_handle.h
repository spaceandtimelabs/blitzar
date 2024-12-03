#pragma once

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// pinned_memory_handle 
//--------------------------------------------------------------------------------------------------
struct pinned_memory_handle {
  void* ptr = nullptr;
  void* next = nullptr;
};
} // namespace sxt::basdv
