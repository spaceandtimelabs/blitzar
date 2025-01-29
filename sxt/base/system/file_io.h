#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"

namespace sxt::bassy {
//--------------------------------------------------------------------------------------------------
// write_to_file
//--------------------------------------------------------------------------------------------------
void write_to_file(const char* filename, basct::cspan<uint8_t> bytes) noexcept;

template <class T>
void write_to_file(const char* filename, basct::cspan<T> values) noexcept {
  basct::cspan<uint8_t> bytes{reinterpret_cast<const uint8_t*>(values.data()),
                              values.size() * sizeof(T)};
  write_to_file(filename, bytes);
}
} // namespace sxt::bassy
