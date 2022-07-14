#pragma once

#include <cstddef>

namespace sxt::mtxi { class index_table; }

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// normalize_product_table
//--------------------------------------------------------------------------------------------------
void normalize_product_table(mtxi::index_table& products,
                             size_t num_entries) noexcept;
}  // namespace sxt::mtxpmp
