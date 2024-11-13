#include "sxt/proof/sumcheck/reduction_gpu.h"

#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

TEST_CASE("we can reduce sumcheck polynomials") {
  std::vector<s25t::element> p;
  std::pmr::vector<s25t::element> partial_terms{memr::get_managed_device_resource()};

  basdv::stream stream;

  SECTION("we can reduce a sime with a single term") {
    p.resize(1);
    partial_terms = {0x123_s25};
    auto fut = reduce_sums(p, stream, partial_terms);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(p[0] == 0x123_s25);
  }
}
