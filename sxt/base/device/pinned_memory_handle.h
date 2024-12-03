#pragma once

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// pinned_memory_handle 
//--------------------------------------------------------------------------------------------------
struct pinned_memory_handle {
  void* ptr = nullptr;
  pinned_memory_handle* next = nullptr;
};
} // namespace sxt::basdv
