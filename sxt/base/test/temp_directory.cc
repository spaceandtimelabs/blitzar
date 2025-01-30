#include "sxt/base/test/temp_directory.h"

#include <filesystem>
#include <cstdio>

#include "sxt/base/error/panic.h"

namespace sxt::bastst {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
temp_directory::temp_directory() noexcept try : name_{std::tmpnam(nullptr)} {
  std::filesystem::create_directory(name_);
} catch (const std::exception& e) {
  baser::panic("failed to create directory {}: {}", name_, e.what());
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
temp_directory::~temp_directory() noexcept {
  try {
  std::filesystem::remove_all(name_);
  } catch(const std::exception& e) {
  baser::panic("failed to remove directory {}: {}", name_, e.what());
  }
}
} // namespace sxt::bastst
