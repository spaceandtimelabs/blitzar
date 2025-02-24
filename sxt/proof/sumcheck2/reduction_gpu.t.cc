#include "sxt/proof/sumcheck2/reduction_gpu.h"

#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk2;
using s25t::operator""_s25;

TEST_CASE("we can reduce sumcheck polynomials") {
}
