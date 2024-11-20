#include "sxt/proof/sumcheck/sum_gpu.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/proof/sumcheck/device_cache.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"
using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

TEST_CASE("we can sum MLEs") {
  std::vector<std::pair<s25t::element, unsigned>> product_table;
  std::vector<unsigned> product_terms;
  std::vector<s25t::element> mles;
  std::vector<s25t::element> p(2);

  SECTION("we can sum an MLE with a single term and n=1") {
    product_table = {{0x1_s25, 1}};
    product_terms = {0};
    device_cache cache{product_table, product_terms};
    mles = {0x123_s25};
    auto fut = sum_gpu(p, cache, mles, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == -mles[0]);
  }

  SECTION("we can sum an MLE with a single term and n=2") {
    product_table = {{0x1_s25, 1}};
    product_terms = {0};
    device_cache cache{product_table, product_terms};
    mles = {0x123_s25, 0x456_s25};
    auto fut = sum_gpu(p, cache, mles, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == mles[1] - mles[0]);
  }
}
