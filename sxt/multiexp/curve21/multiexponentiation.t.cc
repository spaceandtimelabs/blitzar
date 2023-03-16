#include "sxt/multiexp/curve21/multiexponentiation.h"

#include <algorithm>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/test/multiexponentiation.h"

using namespace sxt;
using namespace sxt::mtxc21;

TEST_CASE("we can compute multiexponentiations") {
  auto f = [](basct::cspan<c21t::element_p3> generators,
              basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
    return compute_multiexponentiation(generators, exponents);
  };
  std::mt19937 rng{97834978};
  mtxtst::exercise_multiexponentiation_fn(rng, f);
}

TEST_CASE("we can compute async multiexponentiations") {
  auto f = [](basct::cspan<c21t::element_p3> generators,
              basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
    auto fut = async_compute_multiexponentiation(generators, exponents);
    xens::get_scheduler().run();
    return memmg::managed_array<c21t::element_p3>{std::move(fut.value())};
  };
  std::mt19937 rng{893345};
  mtxtst::exercise_multiexponentiation_fn(rng, f);
}
