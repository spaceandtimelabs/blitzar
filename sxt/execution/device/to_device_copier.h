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

   template <class Cont>
     requires requires(Cont& dst) {
       dst.data();
       dst.size();
     }
   to_device_copier(Cont& dst, basdv::stream& stream) noexcept
       : to_device_copier{basct::span<std::byte>{reinterpret_cast<std::byte*>(dst.data()),
                                                 dst.size() * sizeof(*dst.data())},
                          stream} {}

   xena::future<> copy(basct::cspan<std::byte> src) noexcept;

   template <class Cont>
   xena::future<> copy(const Cont& src) noexcept {
     return this->copy(basct::cspan<std::byte>{reinterpret_cast<const std::byte*>(src.data()),
                                               src.size() * sizeof(*src.data())});
   }

 private:
   basct::span<std::byte> dst_;
   basdv::stream& stream_;
   basdv::pinned_buffer2 active_buffer_;
   basdv::pinned_buffer2 alt_buffer_;
};
} // sxt::xendv
