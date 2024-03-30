#pragma once

#include <cerrno>
#include <cstring>
#include <fstream>
#include <string_view>

#include "sxt/base/error/assert.h"
#include "sxt/base/error/panic.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// in_memory_partition_table_accessor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
class in_memory_partition_table_accessor final : public partition_table_accessor<T> {
 public:
   explicit in_memory_partition_table_accessor(std::string_view filename) noexcept
       : table_{memr::get_pinned_resource()} {
     std::ifstream in{filename, std::ios::binary};
     if (!in.good()) {
       baser::panic("failed to open {}: {}", filename, std::strerror(errno));
     }
     auto pos = in.tellg();
     in.seekg(0, std::ios::end);
     auto size = in.tellg() - pos;
     in.seekg(pos);
     SXT_RELEASE_ASSERT(size % sizeof(T) == 0);
     table_.resize(size / sizeof(T));
     in.read(reinterpret_cast<char*>(table_.data()), size);
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
