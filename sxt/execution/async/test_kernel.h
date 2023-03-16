#pragma once

#include <cstdint>

#include "sxt/base/type/raw_stream.h"

namespace sxt::xenb {
class stream;
}

namespace sxt::xena {
//--------------------------------------------------------------------------------------------------
// add_for_testing
//--------------------------------------------------------------------------------------------------
void add_for_testing(uint64_t* c, bast::raw_stream_t stream, const uint64_t* a, const uint64_t* b,
                     int n) noexcept;
} // namespace sxt::xena
