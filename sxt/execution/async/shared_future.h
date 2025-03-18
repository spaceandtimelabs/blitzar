#pragma once

#include "sxt/execution/async/future.h"

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// shared_future
//--------------------------------------------------------------------------------------------------
template <class T>
class shared_future {
 public:
   shared_future() noexcept = default;

   future<T> get_future() const noexcept {
      return {};
   }
 private:
};
} // namespace sxt::xena
