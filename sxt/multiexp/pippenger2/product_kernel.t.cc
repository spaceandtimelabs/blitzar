#include "sxt/multiexp/pippenger2/product_kernel.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute products with a lookup table") {
  std::pmr::vector<bascrv::element97> products{memr::get_managed_device_resource()};
  std::pmr::vector<bascrv::element97> table{memr::get_managed_device_resource()};
  std::pmr::vector<uint16_t> bitsets{memr::get_managed_device_resource()};

  (void)products;
  (void)table;
  (void)bitsets;
/* template <unsigned ItemsPerThreadLg2, bascrv::element T> */
/* __global__ void product_kernel(T* __restrict__ products, const uint16_t* __restrict__ bitsets, */
/*                                const T* __restrict__ table, unsigned num_products, */
/*                                unsigned n) { */
}
