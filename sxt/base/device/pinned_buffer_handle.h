#pragma once

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// pinned_buffer_handle 
//--------------------------------------------------------------------------------------------------
struct pinned_buffer_handle {
  void* ptr = nullptr;
  pinned_buffer_handle* next = nullptr;
};
} // namespace sxt::basdv
