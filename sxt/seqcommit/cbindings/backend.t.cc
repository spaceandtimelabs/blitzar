#include "sxt/seqcommit/cbindings/backend.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::sqccb;

TEST_CASE("We can verify that invalid backends will error out") {
  const sxt_config config = {-1, 0};
  REQUIRE(sxt_init(&config) != 0);
  REQUIRE(is_backend_initialized() == false);
}

static void test_backend_initialization(int backend) {
  SECTION("The backend is not initialized before calling `sxt_init` function") {
    REQUIRE(is_backend_initialized() == false);
  }

  SECTION("The backend is initialized after calling `sxt_init` function with zero precomputed "
          "elements") {
    const sxt_config config = {backend, 0};
    REQUIRE(sxt_init(&config) == 0);
    REQUIRE(is_backend_initialized() == true);
    reset_backend_for_testing();
  }

  SECTION("The backend is initialized after calling `sxt_init` function with non-zero precomputed "
          "elements") {
    const sxt_config config = {backend, 10};
    REQUIRE(sxt_init(&config) == 0);
    REQUIRE(is_backend_initialized() == true);
    reset_backend_for_testing();
  }

  SECTION("The backend is not initialized after calling the reset function") {
    const sxt_config config = {backend, 0};
    REQUIRE(sxt_init(&config) == 0);
    reset_backend_for_testing();
    REQUIRE(is_backend_initialized() == false);
  }
}

TEST_CASE("We can correctly initialize the naive cpu backend") {
  test_backend_initialization(SXT_NAIVE_BACKEND_CPU);
}

TEST_CASE("We can correctly initialize the naive gpu backend") {
  test_backend_initialization(SXT_NAIVE_BACKEND_GPU);
}

TEST_CASE("We can correctly initialize the pippenger cpu backend") {
  test_backend_initialization(SXT_PIPPENGER_BACKEND_CPU);
}
