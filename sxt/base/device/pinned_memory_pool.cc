#include "sxt/base/device/pinned_memory_pool.h"

#include <cuda_runtime.h>

#include <cassert>

#include "sxt/base/device/pinned_memory_handle.h"
#include "sxt/base/error/panic.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// new_handle 
//--------------------------------------------------------------------------------------------------
static pinned_memory_handle* new_handle() noexcept {
  auto res = new pinned_memory_handle{};
  auto rcode = cudaMallocHost(&res->ptr, pinned_memory_size);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMallocHost failed: {}", cudaGetErrorString(rcode));
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
pinned_memory_pool::pinned_memory_pool(size_t initial_size) noexcept {
  for (size_t i=0; i<initial_size; ++i) {
    auto h = new_handle();
    this->release_handle(h);
  }
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
pinned_memory_pool::~pinned_memory_pool() noexcept {
  while (head_ != nullptr) {
    auto rcode = cudaFreeHost(head_->ptr);
    if (rcode != cudaSuccess) {
      baser::panic("cudaFreeHost failed: {}", cudaGetErrorString(rcode));
    }
    auto next = head_->next;
    delete head_;
    head_ = next;
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
