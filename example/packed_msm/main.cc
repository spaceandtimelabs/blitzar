#include <print>
#include <iostream>

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
  if (false) {
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
