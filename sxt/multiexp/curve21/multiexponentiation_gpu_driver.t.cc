#include "sxt/multiexp/curve21/multiexponentiation_gpu_driver.h"

#include <algorithm>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/multiexp/pippenger/multiexponentiation.h"
#include "sxt/multiexp/test/multiexponentiation.h"

using namespace sxt;
using namespace sxt::mtxc21;

TEST_CASE("we can compute multiexponentiations") {
  multiexponentiation_gpu_driver drv;

  auto f = [&](basct::cspan<c21t::element_p3> generators,
               basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
    memmg::managed_array<c21t::element_p3> generators_p{
        generators.size(),
        memr::get_managed_device_resource(),
    };
    std::copy(generators.begin(), generators.end(), generators_p.begin());
    return mtxpi::compute_multiexponentiation(drv, generators_p, exponents)
        .await_result()
        .as_array<c21t::element_p3>();
  };
  std::mt19937 rng{23151};
  mtxtst::exercise_multiexponentiation_fn(rng, f);
}
