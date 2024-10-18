/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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
#include "cbindings/fixed_pedersen.h"

#include <memory>

#include "cbindings/backend.h"
#include "sxt/cbindings/base/multiexp_handle.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// sxt_multiexp_handle_new
//--------------------------------------------------------------------------------------------------
struct sxt_multiexp_handle* sxt_multiexp_handle_new(unsigned curve_id, const void* generators,
                                                    unsigned n) {
  auto res = std::make_unique<cbnb::multiexp_handle>();
  res->curve_id = static_cast<cbnb::curve_id_t>(curve_id);
  auto backend = cbn::get_backend();
  res->partition_table_accessor =
      backend->make_partition_table_accessor(res->curve_id, generators, n);
  return reinterpret_cast<sxt_multiexp_handle*>(res.release());
}

//--------------------------------------------------------------------------------------------------
// sxt_multiexp_handle_new_from_file
//--------------------------------------------------------------------------------------------------
struct sxt_multiexp_handle* sxt_multiexp_handle_new_from_file(unsigned curve_id,
                                                              const char* filename) {
  auto res = std::make_unique<cbnb::multiexp_handle>();
  res->curve_id = static_cast<cbnb::curve_id_t>(curve_id);
  auto backend = cbn::get_backend();
  res->partition_table_accessor = backend->read_partition_table_accessor(res->curve_id, filename);
  return reinterpret_cast<sxt_multiexp_handle*>(res.release());
}

//--------------------------------------------------------------------------------------------------
// sxt_multiexp_handle_free
//--------------------------------------------------------------------------------------------------
void sxt_multiexp_handle_free(struct sxt_multiexp_handle* handle) {
  delete reinterpret_cast<cbnb::multiexp_handle*>(handle);
}

//--------------------------------------------------------------------------------------------------
// sxt_fixed_multiexponentiation
//--------------------------------------------------------------------------------------------------
void sxt_fixed_multiexponentiation(void* res, const struct sxt_multiexp_handle* handle,
                                   unsigned element_num_bytes, unsigned num_outputs, unsigned n,
                                   const uint8_t* scalars) {
  auto backend = cbn::get_backend();
  auto h = reinterpret_cast<const cbnb::multiexp_handle*>(handle);
  backend->fixed_multiexponentiation(res, h->curve_id, *h->partition_table_accessor,
                                     element_num_bytes, num_outputs, n, scalars);
}

//--------------------------------------------------------------------------------------------------
// sxt_fixed_packed_multiexponentiation
//--------------------------------------------------------------------------------------------------
void sxt_fixed_packed_multiexponentiation(void* res, const struct sxt_multiexp_handle* handle,
                                          const unsigned* output_bit_table, unsigned num_outputs,
                                          unsigned n, const uint8_t* scalars) {
  auto backend = cbn::get_backend();
  auto h = reinterpret_cast<const cbnb::multiexp_handle*>(handle);
  backend->fixed_multiexponentiation(res, h->curve_id, *h->partition_table_accessor,
                                     output_bit_table, num_outputs, n, scalars);
}

//--------------------------------------------------------------------------------------------------
// sxt_fixed_vlen_multiexponentiation
//--------------------------------------------------------------------------------------------------
void sxt_fixed_vlen_multiexponentiation(void* res, const struct sxt_multiexp_handle* handle,
                                        const unsigned* output_bit_table,
                                        const unsigned* output_lengths, unsigned num_outputs,
                                        const uint8_t* scalars) {
  auto backend = cbn::get_backend();
  auto h = reinterpret_cast<const cbnb::multiexp_handle*>(handle);
  backend->fixed_multiexponentiation(res, h->curve_id, *h->partition_table_accessor,
                                     output_bit_table, output_lengths, num_outputs, scalars);
}
