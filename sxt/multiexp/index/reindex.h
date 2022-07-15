#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"
#include "sxt/base/functional/function_ref.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// reindex_rows
//--------------------------------------------------------------------------------------------------
void reindex_rows(basct::span<basct::span<uint64_t>> rows, basct::span<uint64_t>& values) noexcept;

void reindex_rows(basct::span<basct::span<uint64_t>> rows, basct::span<uint64_t>& values,
                  basf::function_ref<size_t(basct::cspan<uint64_t>)> offset_functor) noexcept;
} // namespace sxt::mtxi
