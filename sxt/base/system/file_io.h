#pragma once

#include <cerrno>
#include <cstddef>
#include <fstream>

#include "sxt/base/container/span.h"
#include "sxt/base/error/panic.h"

namespace sxt::bassy {
//--------------------------------------------------------------------------------------------------
// file_size
//--------------------------------------------------------------------------------------------------
size_t file_size(const char* filename) noexcept;

//--------------------------------------------------------------------------------------------------
// write_file
//--------------------------------------------------------------------------------------------------
void write_file(const char* filename, basct::cspan<uint8_t> bytes) noexcept;

template <class T>
void write_file(const char* filename, basct::cspan<T> values) noexcept {
  basct::cspan<uint8_t> bytes{reinterpret_cast<const uint8_t*>(values.data()),
                              values.size() * sizeof(T)};
  write_file(filename, bytes);
}

//--------------------------------------------------------------------------------------------------
// read_file
//--------------------------------------------------------------------------------------------------
template <class T> void read_file(std::vector<T>& values, const char* filename) noexcept {
  auto sz = file_size(filename);
  if (sz % sizeof(T) != 0) {
    baser::panic("{} file size {} is not a multiple of element size {}", filename, sz, sizeof(T));
  }
  auto n = sz / sizeof(T);
  std::ifstream in{filename, std::ios::binary};
  if (!in.good()) {
    baser::panic("failed to open {}: {}", filename, std::strerror(errno));
  }
  values.resize(n);
  in.read(reinterpret_cast<char*>(values.data()), sz);
  if (!in.good()) {
    baser::panic("failed to read {}: {}", filename, std::strerror(errno));
  }
}
} // namespace sxt::bassy
