#include <print>

#include "sxt/multiexp/pippenger2/multiexponentiation_serialization.h"
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
using namespace sxt;

int main() {
  mtxpp2::packed_multiexponentiation_descriptor<cg1t::element_p2, cg1t::compact_element> descr;
  mtxpp2::read_multiexponentiation(descr, "/home/rnburn/proj/blitzar/example/packed_msm/"
                                          "dory_multi_gpu_bug/packed-multiexponentiation-0/");
  /* f(std::type_identity<cg1t::compact_element>{}, std::type_identity<cg1t::element_p2>{}); */
  std::println("arf");
/* template <bascrv::element T, class U> */
/* void read_multiexponentiation(packed_multiexponentiation_descriptor<T, U>& descr, */
/*                               std::string_view dir) noexcept { */
  return 0;
}
