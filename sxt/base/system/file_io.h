#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"

namespace sxt::bassy {
//--------------------------------------------------------------------------------------------------
// write_bytes
//--------------------------------------------------------------------------------------------------
void write_bytes(const char* filename, basct::cspan<uint8_t> bytes) noexcept;
} // namespace sxt::bassy
