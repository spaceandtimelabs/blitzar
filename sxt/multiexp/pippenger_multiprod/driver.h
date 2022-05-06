#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::mtxi { struct clump2_descriptor; }

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// driver
//--------------------------------------------------------------------------------------------------
class driver {
 public:
   virtual ~driver() noexcept = default;

   virtual void apply_partition_operation(
       memmg::managed_array<void>& inout,
       basct::cspan<uint64_t> partition_markers,
       size_t partition_size) const noexcept = 0;

   virtual void apply_clump2_operation(
       memmg::managed_array<void>& inout, basct::cspan<uint64_t> markers,
       const mtxi::clump2_descriptor& descriptor) const noexcept = 0;
};
} // namespace sxt::mtxpmp
