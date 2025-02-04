/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include <iostream>
#include <print>

#include "sxt/curve_g1/operation/add.h"
#include "sxt/curve_g1/operation/compression.h"
#include "sxt/curve_g1/operation/double.h"
#include "sxt/curve_g1/operation/neg.h"
#include "sxt/curve_g1/type/compressed_element.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/curve_gk/operation/add.h"
#include "sxt/curve_gk/operation/double.h"
#include "sxt/curve_gk/operation/neg.h"
#include "sxt/curve_gk/type/conversion_utility.h"
#include "sxt/curve_gk/type/element_affine.h"
#include "sxt/curve_gk/type/element_p2.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/multiexp/pippenger2/multiexponentiation.h"
#include "sxt/multiexp/pippenger2/multiexponentiation_serialization.h"

using namespace sxt;

int main() {
  mtxpp2::packed_multiexponentiation_descriptor<cg1t::element_p2, cg1t::compact_element> descr;
  mtxpp2::read_multiexponentiation(descr, "/home/rnburn/proj/blitzar/example/packed_msm/"
                                          "dory_multi_gpu_bug/packed-multiexponentiation-0/");
  std::println("num_outputs = {}", descr.output_bit_table.size());
  std::vector<cg1t::element_p2> res(descr.output_bit_table.size());
  for (auto oi : descr.output_bit_table) {
    std::println("tbl: {}", oi);
  }
  if (true) {
    auto fut = mtxpp2::async_multiexponentiate<cg1t::element_p2>(
        res, *descr.accessor, descr.output_bit_table, descr.scalars);
    xens::get_scheduler().run();
  } else {
    mtxpp2::multiexponentiate<cg1t::element_p2>(res, *descr.accessor, descr.output_bit_table,
                                                descr.scalars);
  }
  for (auto& ri : res) {
    std::cout << ri.X << " " << ri.Y << "\n";
  }
  return 0;
}
