#include "sxt/base/error/panic.h"

#include <cstdlib>
#include <iostream>

namespace sxt::baser {
//--------------------------------------------------------------------------------------------------
// panic
//--------------------------------------------------------------------------------------------------
[[noreturn]] void panic(std::string_view message, int line, const char* file) noexcept {
  std::cerr << file << ":" << line << " panic: " << message << "\n";
  std::abort();
}
} // namespace sxt::baser
