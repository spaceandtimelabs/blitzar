/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sxt/multiexp/pippenger_multiprod/product_table_normalization.h"

#include <algorithm>

#include "sxt/multiexp/index/index_table.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// normalize_product_table
//--------------------------------------------------------------------------------------------------
void normalize_product_table(mtxi::index_table& products, size_t num_entries) noexcept {
  size_t num_entries_p = num_entries + products.num_rows() * 2;
  mtxi::index_table products_p{products.num_rows(), num_entries_p};
  auto entry_data = products_p.entry_data();
  auto rows = products.cheader();
  auto rows_p = products_p.header();
  for (size_t row_index = 0; row_index < rows_p.size(); ++row_index) {
    auto row = rows[row_index];
    rows_p[row_index] = {entry_data, row.size() + 2};
    *entry_data++ = row_index;
    *entry_data++ = 0;
    entry_data = std::copy(row.begin(), row.end(), entry_data);
  }
  products = std::move(products_p);
}
} // namespace sxt::mtxpmp
