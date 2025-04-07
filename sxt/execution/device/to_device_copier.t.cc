#include "sxt/execution/device/to_device_copier.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::xendv;

TEST_CASE("we can copy memory from host to device") {
  memmg::managed_array<int> dev{memr::get_managed_device_resource()};
  std::vector<int> host;
  basdv::stream stream;

  SECTION("we can copy an empty array") {
    to_device_copier copier{dev, stream};
    auto fut = copier.copy(host);
    REQUIRE(fut.ready());
  }
}
