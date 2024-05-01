#pragma once

#include <string_view>

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// partition_table_accessor_base
//--------------------------------------------------------------------------------------------------
class partition_table_accessor_base {
 public:
   virtual ~partition_table_accessor_base() noexcept = default;

  virtual void write_to_file(std::string_view filename) const noexcept = 0;
};
} // namespace sxt::mtxpp2
