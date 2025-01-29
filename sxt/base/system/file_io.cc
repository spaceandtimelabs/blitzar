#include "sxt/base/system/file_io.h"

#include <fstream>
#include <cerrno>

#include "sxt/base/error/panic.h"

namespace sxt::bassy {
//--------------------------------------------------------------------------------------------------
// write_to_file
//--------------------------------------------------------------------------------------------------
void write_to_file(const char* filename, basct::cspan<uint8_t> bytes) noexcept {
  std::ofstream out{filename, std::ios::binary};
  if (!out.good()) {
    baser::panic("failed to open {}: {}", filename, std::strerror(errno));
  }
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  out.close();
  if (!out.good()) {
    baser::panic("failed to close {}: {}", filename, std::strerror(errno));
  }
}
} // namespace sxt::bassy
