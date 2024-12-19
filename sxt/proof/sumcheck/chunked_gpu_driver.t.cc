#include "sxt/proof/sumcheck/chunked_gpu_driver.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/proof/sumcheck/driver_test.h"
using namespace sxt;
using namespace sxt::prfsk;

TEST_CASE("we can perform the primitive operations for sumcheck proofs") {
  SECTION("we handle the case when only chunking is used") {
    chunked_gpu_driver drv{0.0};
    exercise_driver(drv);
  }

  SECTION("we handle the case when the chunked driver falls back to the single gpu driver") {
    chunked_gpu_driver drv{1.0};
    exercise_driver(drv);
  }
}
