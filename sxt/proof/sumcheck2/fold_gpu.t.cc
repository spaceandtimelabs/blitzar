#include "sxt/proof/sumcheck2/fold_gpu.h"

#include <vector>

#include "sxt/base/iterator/split.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/realization/field.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk2;
using s25t::operator""_s25;

TEST_CASE("we can fold scalars using the gpu") {
  using T = s25t::element;
  std::vector<s25t::element> mles, mles_p, expected;

  auto r = 0xabc123_s25;
  auto one_m_r = 0x1_s25 - r;

  SECTION("we can fold a single mle with n=2") {
    mles = {0x1_s25, 0x2_s25};
    mles_p.resize(1);
    auto fut = fold_gpu<T>(mles_p, mles, 2, r);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected = {
        one_m_r * mles[0] + r * mles[1],
    };
    REQUIRE(mles_p == expected);
  }

  SECTION("we can fold a single mle with n=3") {
    mles = {0x123_s25, 0x456_s25, 0x789_s25};
    mles_p.resize(2);
    auto fut = fold_gpu<T>(mles_p, mles, 3, r);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected = {
        one_m_r * mles[0] + r * mles[2],
        one_m_r * mles[1],
    };
    REQUIRE(mles_p == expected);
  }

  SECTION("we can split folds") {
    basit::split_options split_options{
        .min_chunk_size = 1,
        .max_chunk_size = 1,
        .split_factor = 2,
    };
    mles = {0x123_s25, 0x456_s25, 0x789_s25, 0x101112_s25};
    mles_p.resize(2);
    auto fut = fold_gpu<T>(mles_p, split_options, mles, 4, r);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected = {
        one_m_r * mles[0] + r * mles[2],
        one_m_r * mles[1] + r * mles[3],
    };
    REQUIRE(mles_p == expected);
  }
}
