#include "sxt/base/device/event.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("basdv::event provides an RAII wrapper around CUDA events") {
  event ev;

  SECTION("we can move-construct events") {
    bast::raw_cuda_event_t ptr = ev;
    event ev2{std::move(ev)};
    REQUIRE(ev2 == ptr);
    REQUIRE(static_cast<bast::raw_cuda_event_t>(ev) == nullptr);
  }

  SECTION("we can move-assign an event") {
    bast::raw_cuda_event_t ptr = ev;
    event ev2;
    ev2 = std::move(ev);
    REQUIRE(ev2 == ptr);
    REQUIRE(static_cast<bast::raw_cuda_event_t>(ev) == nullptr);
  }
}
