/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
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

#include "sxt/cbindings/backend/computational_backend.h"

#include "sxt/cbindings/base/curve_id_utility.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor.h"

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// read_partition_table_accessor
//--------------------------------------------------------------------------------------------------
std::unique_ptr<mtxpp2::partition_table_accessor_base>
computational_backend::read_partition_table_accessor(cbnb::curve_id_t curve_id,
                                                     const char* filename) const noexcept {
  std::unique_ptr<mtxpp2::partition_table_accessor_base> res;
  cbnb::switch_curve_type(
      curve_id, [&]<class U, class T>(std::type_identity<U>, std::type_identity<T>) noexcept {
        res = std::make_unique<mtxpp2::in_memory_partition_table_accessor<U>>(filename);
      });
  return res;
}

//--------------------------------------------------------------------------------------------------
// write_partition_table_accessor
//--------------------------------------------------------------------------------------------------
void computational_backend::write_partition_table_accessor(
    cbnb::curve_id_t curve_id, const mtxpp2::partition_table_accessor_base& accessor,
    const char* filename) const noexcept {
  cbnb::switch_curve_type(
      curve_id, [&]<class U, class T>(std::type_identity<U>, std::type_identity<T>) noexcept {
        static_cast<const mtxpp2::partition_table_accessor<U>&>(accessor).write_to_file(filename);
      });
}
} // namespace sxt::cbnbck
