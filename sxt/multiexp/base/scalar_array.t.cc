#include "sxt/multiexp/base/scalar_array.h"

#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxb;

TEST_CASE("we can copy scalars to a device array") {
  basdv::stream stream;
  memmg::managed_array<uint8_t> array{memr::get_managed_device_resource()};

  SECTION("we can copy multiple scalar arrays to the device") {
    std::vector<uint8_t> scalars1 = {1, 2, 3, 4};
    std::vector<uint8_t> scalars2 = {5, 6, 7, 8};
    array.resize(8);
    std::vector<const uint8_t*> scalars = {scalars1.data(), scalars2.data()};
    sxt::mtxb::make_device_scalar_array(array, stream, scalars, 2, 2);
    basdv::synchronize_stream(stream);
    memmg::managed_array<uint8_t> expected = {1, 2, 3, 4, 5, 6, 7, 8};
    REQUIRE(array == expected);
  }
}