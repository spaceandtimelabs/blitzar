#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// add_ints
//--------------------------------------------------------------------------------------------------
void add_ints(basct::span<uint64_t> result, basct::cspan<basct::cspan<uint64_t>> terms,
              basct::cspan<uint64_t> inputs) noexcept;
} // namespace sxt::mtxtst
