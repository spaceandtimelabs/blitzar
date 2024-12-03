#include "sxt/base/device/pinned_memory_pool.h"

#include <cassert>

#include "sxt/base/device/pinned_memory_handle.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// new_handle 
//--------------------------------------------------------------------------------------------------
static pinned_memory_handle* new_handle() noexcept {
  return nullptr;
}

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
pinned_memory_pool::pinned_memory_pool(size_t initial_size) noexcept {
  (void)initial_size;
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
pinned_memory_pool::~pinned_memory_pool() noexcept {
  while (head_ != nullptr) {
    this->release_handle(head_);
  }
}

//--------------------------------------------------------------------------------------------------
// aquire_handle
//--------------------------------------------------------------------------------------------------
pinned_memory_handle* pinned_memory_pool::aquire_handle() noexcept {
  if (head_ == nullptr) {
    head_ = new_handle();
  }
  auto res = head_;
  head_ = res->next;
  res->next = nullptr;
  return res;
}

//--------------------------------------------------------------------------------------------------
// release_handle
//--------------------------------------------------------------------------------------------------
void pinned_memory_pool::release_handle(pinned_memory_handle* handle) noexcept { 
  assert(handle->next == nullptr);
  handle->next = head_;
  head_ = handle;
}
} // namespace sxt::basdv
