#include "sxt/base/device/event.h"

#include <cuda_runtime.h>

#include <string>
#include <utility>

#include "sxt/base/error/panic.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
event::event() noexcept {
  auto rcode = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
  if (rcode != cudaSuccess) {
    baser::panic("cudaEventCreate failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

event::event(event&& other) noexcept : event_{std::exchange(other.event_, nullptr)} {}

//--------------------------------------------------------------------------------------------------
// assignment
//--------------------------------------------------------------------------------------------------
event& event::operator=(event&& other) noexcept {
  this->clear();
  event_ = std::exchange(other.event_, nullptr);
  return *this;
}

//--------------------------------------------------------------------------------------------------
// clear
//--------------------------------------------------------------------------------------------------
void event::clear() noexcept {
  if (event_ == nullptr) {
    return;
  }
  auto rcode = cudaEventDestroy(event_);
  if (rcode != cudaSuccess) {
    baser::panic("cudaEventDestroy failed: " + std::string(cudaGetErrorString(rcode)));
  }
  event_ = nullptr;
}

//--------------------------------------------------------------------------------------------------
// query_is_ready
//--------------------------------------------------------------------------------------------------
bool event::query_is_ready() noexcept {
  auto rcode = cudaEventQuery(event_);
  if (rcode == cudaSuccess) {
    return true;
  }
  if (rcode == cudaErrorNotReady) {
    return false;
  }
  baser::panic("cudaEventQuery failed: " + std::string(cudaGetErrorString(rcode)));
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
event::~event() noexcept { this->clear(); }
} // namespace sxt::basdv
