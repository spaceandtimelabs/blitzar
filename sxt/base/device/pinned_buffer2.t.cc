#include "sxt/base/device/pinned_buffer2.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("we can manage a buffer of pinned memory") {
  SECTION("we can construct and deconstruct a buffer") {
    pinned_buffer2 buf;
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.empty());
  }

  SECTION("we can add a single byte to a buffer") {
    pinned_buffer2 buf;
    std::vector<std::byte> data = {static_cast<std::byte>(123)};
    auto rest = buf.fill_from_host(data);
    REQUIRE(rest.empty());
    REQUIRE(buf.size() == 1);
    REQUIRE(*static_cast<std::byte*>(buf.data()) == data[0]);
  }

  SECTION("we can fill a buffer") {
    pinned_buffer2 buf;
    std::vector<std::byte> data(buf.capacity() + 1, static_cast<std::byte>(123));
    auto rest = buf.fill_from_host(data);
    REQUIRE(rest.size() == 1);
    REQUIRE(buf.size() == buf.capacity());
    REQUIRE(buf.full());
  }
}
