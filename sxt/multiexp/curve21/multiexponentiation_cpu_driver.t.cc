#include "sxt/multiexp/curve21/multiexponentiation_cpu_driver.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/curve21/naive_multiproduct_solver.h"
#include "sxt/multiexp/pippenger/multiexponentiation.h"
#include "sxt/multiexp/test/multiexponentiation.h"

using namespace sxt;
using namespace sxt::mtxc21;

TEST_CASE("we can compute multiexponentiations") {
  naive_multiproduct_solver solver;
  multiexponentiation_cpu_driver drv{&solver};
  auto f = [&](basct::cspan<c21t::element_p3> generators,
               basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
    return mtxpi::compute_multiexponentiation(drv, generators, exponents)
        .value()
        .as_array<c21t::element_p3>();
  };
  std::mt19937 rng{9873324};
  mtxtst::exercise_multiexponentiation_fn(rng, f);
}
