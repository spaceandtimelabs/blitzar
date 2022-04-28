#pragma once

#include <cstddef>

#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::mtxi { class index_table; }

namespace sxt::mtxpmp {
class driver;

//--------------------------------------------------------------------------------------------------
// partition_inputs
//--------------------------------------------------------------------------------------------------
void partition_inputs(memmg::managed_array<void>& inputs,
                      mtxi::index_table& product_table, const driver& drv,
                      size_t partition_size) noexcept;
} // namespace sxt::mtxpmp
