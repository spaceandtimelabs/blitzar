#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"
#include "sxt/base/functional/function_ref.h"

namespace sxt::mtxi {
class index_table;

//--------------------------------------------------------------------------------------------------
// transpose
//--------------------------------------------------------------------------------------------------
size_t transpose(
    index_table& table, basct::cspan<basct::cspan<uint64_t>> rows,
    size_t distinct_entry_count, size_t padding,
    basf::function_ref<size_t(basct::cspan<uint64_t>)> offset_functor) noexcept;

size_t transpose(index_table& table, basct::cspan<basct::cspan<uint64_t>> rows,
                 size_t distinct_entry_count, size_t padding = 0) noexcept;
}  // namespace sxt::mtxi
