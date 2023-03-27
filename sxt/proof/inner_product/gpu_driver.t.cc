#include "sxt/proof/inner_product/gpu_driver.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/proof/inner_product/driver_test.h"

using namespace sxt;
using namespace sxt::prfip;

TEST_CASE("gpu_driver provides a backend for inner product proving and verifying") {
  gpu_driver drv;
  exercise_driver(drv);
}
