#pragma once

#include <cerrno>
#include <cstring>
#include <fstream>
#include <string_view>

#include "sxt/base/error/panic.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"
#include "sxt/memory/management/managed_array.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// in_memory_partition_table_accessor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
class in_memory_partition_table_accessor final : public partition_table_accessor<T> {
 public:
   explicit in_memory_partition_table_accessor(std::string_view filename) noexcept {
     std::ifstream in{filename, std::ios::binary};
     if (!in.good()) {
       baser::panic("failed to open {}: {}", filename, std::strerror(errno));
     }
     auto pos = in.tellg();
     in.seekg(0, std::ios::end);
     auto size = in.tellg() - pos;
     in.seekg(pos);
     (void)size;
     (void)filename;
   }

   void async_copy_precomputed_sums_to_device(basct::span<T> dest, bast::raw_stream_t stream,
                                              unsigned first) const noexcept override {
     (void)dest;
     (void)stream;
     (void)first;
   }

 private:
   memmg::managed_array<T> table_;
};
} // namespace sxt::mtxpp2
