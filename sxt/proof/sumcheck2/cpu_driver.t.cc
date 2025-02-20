#include "sxt/proof/sumcheck2/cpu_driver.h"

#include "sxt/proof/sumcheck2/driver_test.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::prfsk2;

TEST_CASE("we can perform the primitive operations for sumcheck proofs") {
  cpu_driver<s25t::element> drv;
  exercise_driver(drv);
}
