#include "sxt/proof/sumcheck2/chunked_gpu_driver.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/proof/sumcheck2/driver_test.h"

using namespace sxt;
using namespace sxt::prfsk2;

TEST_CASE("we can perform the primitive operations for sumcheck proofs") {
  SECTION("we handle the case when only chunking is used") {
    chunked_gpu_driver<s25t::element> drv{0.0};
    exercise_driver(drv);
  }

  SECTION("we handle the case when the chunked driver falls back to the single gpu driver") {
    chunked_gpu_driver<s25t::element> drv{1.0};
    exercise_driver(drv);
  }
}
