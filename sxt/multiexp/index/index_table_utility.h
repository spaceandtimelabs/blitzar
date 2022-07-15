#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"

namespace sxt::mtxi {
class index_table;

//--------------------------------------------------------------------------------------------------
// init_rows
//--------------------------------------------------------------------------------------------------
void init_rows(index_table& table, basct::cspan<size_t> sizes) noexcept;
} // namespace sxt::mtxi
