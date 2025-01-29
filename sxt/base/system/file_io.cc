#include "sxt/base/system/file_io.h"

#include <filesystem>

namespace sxt::bassy {
//--------------------------------------------------------------------------------------------------
// file_size
//--------------------------------------------------------------------------------------------------
size_t file_size(const char* filename) noexcept {
  try {
    return std::filesystem::file_size(filename);
  } catch (const std::exception& e) {
    baser::panic("failed to get file size of {}: {}", filename, e.what());
  }
}

//--------------------------------------------------------------------------------------------------
// write_file
//--------------------------------------------------------------------------------------------------
void write_file(const char* filename, basct::cspan<uint8_t> bytes) noexcept {
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
