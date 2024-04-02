#include "sxt/base/test/temp_file.h"

#include <cerrno>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <print>

namespace sxt::bastst {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
temp_file::temp_file(std::ios_base::openmode openmode) noexcept
    : name_{std::tmpnam(nullptr)}, out_{name_, openmode} {
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
temp_file::~temp_file() noexcept {
  out_.close();
  auto rcode = std::remove(name_.c_str());
  if (rcode != 0) {
    std::println(stderr, "failed to close file {}: {}", name_, std::strerror(errno));
    std::abort();
  }
}
} // namespace sxt::bastst
