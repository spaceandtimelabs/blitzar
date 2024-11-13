#include "sxt/proof/sumcheck/reduction_gpu.h"

#include <vector>

#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
using namespace sxt;
using namespace sxt::prfsk;

TEST_CASE("we can reduce sumcheck polynomials") {
  std::vector<s25t::element> p;
  std::pmr::vector<s25t::element> partial_terms{memr::get_managed_device_resource()};
}
#if 0
xena::future<> reduce_sums(basct::span<s25t::element> p, basdv::stream& stream,
                           basct::cspan<s25t::element> partial_terms) noexcept;
#endif
