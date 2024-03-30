#pragma once

#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

#include "sxt/memory/management/managed_array.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// in_memory_partition_table_accessor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
class in_memory_partition_table_accessor final : public partition_table_accessor<T> {
 public:
   void async_copy_precomputed_sums_to_device(basct::span<T> dest, bast::raw_stream_t stream,
                                              unsigned first) const noexcept {
     (void)dest;
     (void)stream;
     (void)first;
   }

 private:
   memmg::managed_array<T> table_;
};
} // namespace sxt::mtxpp2
