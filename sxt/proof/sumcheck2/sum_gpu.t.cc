#include "sxt/proof/sumcheck2/sum_gpu.h"

#include <vector>

#include "sxt/base/iterator/split.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/proof/sumcheck/device_cache.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/realization/field.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk2;
using s25t::operator""_s25;

#if 0
TEST_CASE("we can sum MLEs") {
  using T = s25t::element;

  std::vector<std::pair<T, unsigned>> product_table;
  std::vector<unsigned> product_terms;
  std::vector<T> mles;
  std::vector<T> p(2);

  SECTION("we can sum an MLE with a single term and n=1") {
    product_table = {{0x1_s25, 1}};
    product_terms = {0};
    device_cache<T> cache{product_table, product_terms};
    mles = {0x123_s25};
    auto fut = sum_gpu<T>(p, cache, mles, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == -mles[0]);
  }

  SECTION("we can sum an MLE with a single term, n=1, and a non-unity multiplier") {
    product_table = {{0x2_s25, 1}};
    product_terms = {0};
    device_cache<T> cache{product_table, product_terms};
    mles = {0x123_s25};
    auto fut = sum_gpu<T>(p, cache, mles, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == product_table[0].first * mles[0]);
    REQUIRE(p[1] == -product_table[0].first * mles[0]);
  }

  SECTION("we can sum an MLE with a single term and n=2") {
    product_table = {{0x1_s25, 1}};
    product_terms = {0};
    device_cache<T> cache{product_table, product_terms};
    mles = {0x123_s25, 0x456_s25};
    auto fut = sum_gpu<T>(p, cache, mles, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == mles[1] - mles[0]);
  }

  SECTION("we can sum an MLE with multiple terms and n=1") {
    p.resize(3);
    product_table = {{0x1_s25, 2}};
    product_terms = {0, 1};
    device_cache<T> cache{product_table, product_terms};
    mles = {0x123_s25, 0x456_s25};
    auto fut = sum_gpu<T>(p, cache, mles, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0] * mles[1]);
    REQUIRE(p[1] == -mles[0] * mles[1] - mles[1] * mles[0]);
    REQUIRE(p[2] == mles[0] * mles[1]);
  }

  SECTION("we can sum multiple mles") {
    product_table = {
        {0x1_s25, 1},
        {0x1_s25, 1},
    };
    product_terms = {0, 1};
    device_cache<T> cache{product_table, product_terms};
    mles = {0x123_s25, 0x456_s25};
    auto fut = sum_gpu<T>(p, cache, mles, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0] + mles[1]);
    REQUIRE(p[1] == -mles[0] - mles[1]);
  }

  SECTION("we can chunk sums with n=4") {
    product_table = {{0x1_s25, 1}};
    product_terms = {0};
    device_cache<T> cache{product_table, product_terms};
    mles = {0x123_s25, 0x456_s25, 0x789_s25, 0x91011_s25};
    basit::split_options options{
        .min_chunk_size = 1,
        .max_chunk_size = 1,
        .split_factor = 2,
    };
    auto fut = sum_gpu<T>(p, cache, options, mles, 4);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0] + mles[1]);
    REQUIRE(p[1] == (mles[2] - mles[0]) + (mles[3] - mles[1]));
  }

  SECTION("we can chunk sums with n=4") {
    product_table = {{0x1_s25, 1}};
    product_terms = {0};
    device_cache<T> cache{product_table, product_terms};
    mles = {0x2_s25, 0x4_s25, 0x7_s25, 0x9_s25};
    basit::split_options options{
        .min_chunk_size = 16,
        .max_chunk_size = 16,
        .split_factor = 2,
    };
    auto fut = sum_gpu<T>(p, cache, options, mles, 4);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == mles[0] + mles[1]);
    REQUIRE(p[1] == (mles[2] - mles[0]) + (mles[3] - mles[1]));
  }
}
#endif
