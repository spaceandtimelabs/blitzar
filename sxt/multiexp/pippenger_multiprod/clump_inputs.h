#pragma once

#include <cstddef>

#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::mtxi { class index_table; }

namespace sxt::mtxpmp {
class driver;

//--------------------------------------------------------------------------------------------------
// clump_inputs
//--------------------------------------------------------------------------------------------------
void clump_inputs(memmg::managed_array<void>& inputs,
                      mtxi::index_table& products, const driver& drv,
                      size_t clump_size) noexcept;
} // namespace sxt::mtxpmp
