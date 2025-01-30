#pragma once

#include <string>

namespace sxt::bastst {
//--------------------------------------------------------------------------------------------------
// temp_directory
//--------------------------------------------------------------------------------------------------
class temp_directory {
public:
  temp_directory() noexcept;

  ~temp_directory() noexcept;

  const std::string& name() const noexcept { return name_; }

public:
  std::string name_;
};
} // namespace sxt::bastst
