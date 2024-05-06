#include "cbindings/fixed_pedersen.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve21/type/literal.h"
using namespace sxt;
using sxt::c21t::operator""_c21;

TEST_CASE("we can compute multi-exponentiations with a fixed set of generators") {
  std::vector<c21t::element_p3> generators = {
      0x123_c21,
  };

  const sxt_config config = {SXT_GPU_BACKEND, 0};
  REQUIRE(sxt_init(&config) == 0);

  SECTION("we can create and free a handle") {
    auto h =
        sxt_multiexp_handle_new(SXT_CURVE_RISTRETTO255, static_cast<void*>(generators.data()), 1);
    sxt_multiexp_handle_free(h);
  }
}
