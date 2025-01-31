#pragma once

#include <atomic>
#include <string_view>
#include <string>

namespace sxt::bassy {
//--------------------------------------------------------------------------------------------------
// directory_recorder
//--------------------------------------------------------------------------------------------------
class directory_recorder {
 public:
   directory_recorder(std::string base_name, std::string_view force_record_dir={}) noexcept;

   bool recording() const noexcept {
     return !name_.empty();
   }

   std::string_view name() const noexcept { return name_; }

 private:
   static std::atomic<unsigned> counter_;
   std::string name_;
};
} // namespace sxt::bassy
