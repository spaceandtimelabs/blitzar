#pragma once

#include <fstream>
#include <string>

namespace sxt::bastst {
//--------------------------------------------------------------------------------------------------
// temp_file 
//--------------------------------------------------------------------------------------------------
/**
 * Set up a temporary file that is deleted in the destructor.
 *
 * This is meant to make easier to write tests involving files.
 */
class temp_file {
 public:
   explicit temp_file(std::ios_base::openmode = std::ios_base::out) noexcept;

   ~temp_file() noexcept;

   const std::string& name() const noexcept { return name_; }

   std::ofstream& stream() noexcept { return out_; }
 private:
   std::string name_;
   std::ofstream out_;
};
} // namespace sxt::bastst
