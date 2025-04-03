#pragma once

#include <optional>

#include "sxt/base/container/span.h"
#include "sxt/base/device/pinned_buffer.h"
#include "sxt/base/device/pinned_buffer2.h"
#include "sxt/base/device/stream.h"
#include "sxt/execution/async/future.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// to_device_copier
//--------------------------------------------------------------------------------------------------
class to_device_copier {
 public:
   to_device_copier(basct::span<std::byte> dst, basdv::stream& stream) noexcept;

   xena::future<> copy(basct::cspan<std::byte> src) noexcept;

 private:
   basct::span<std::byte> dst_;
   basdv::stream& stream_;
   basdv::pinned_buffer2 active_buffer_;
   basdv::pinned_buffer2 alt_buffer_;
};
} // sxt::xendv
