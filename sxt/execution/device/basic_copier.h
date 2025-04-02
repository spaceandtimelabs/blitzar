#pragma once

#include <optional>

#include "sxt/base/container/span.h"
#include "sxt/base/device/pinned_buffer.h"
#include "sxt/execution/async/future.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// basic_copier
//--------------------------------------------------------------------------------------------------
class basic_copier {
 public:

   xena::future<> copy(basct::cspan<std::byte> src) noexcept;

 private:
   basct::span<std::byte> dst_;

   size_t active_count_ = 0;
   std::optional<basdv::pinned_buffer> active_buffer_;

   std::optional<basdv::pinned_buffer> alt_buffer_;
   xena::future<> alt_future_;
};
} // sxt::xendv
