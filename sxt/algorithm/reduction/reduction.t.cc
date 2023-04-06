#include "sxt/algorithm/reduction/reduction.h"

#include <random>

#include "sxt/algorithm/base/identity_mapper.h"
#include "sxt/algorithm/reduction/test_reducer.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::algr;

static memmg::managed_array<uint64_t> make_dataset(std::mt19937& rng, unsigned n) {
  memmg::managed_array<uint64_t> res{n, memr::get_managed_device_resource()};
  for (size_t i = 0; i < n; ++i) {
    res[i] = rng();
  }
  return res;
}

static int64_t compute_sum(const memmg::managed_array<uint64_t>& data) noexcept {
  uint64_t res = 0;
  for (auto x : data) {
    res += x;
  }
  return res;
}

TEST_CASE("we can perform reductions on the GPU") {
  std::mt19937 rng{0};

  basdv::stream stream;

  SECTION("we can reduce a single element") {
    auto data = make_dataset(rng, 1);
    auto res = reduce<test_add_reducer>(std::move(stream), algb::identity_mapper{data.data()}, 1);
    xens::get_scheduler().run();
    basdv::synchronize_device();
    REQUIRE(res.value() == data[0]);
  }

  SECTION("we can reduce two elements") {
    auto data = make_dataset(rng, 2);
    auto res = reduce<test_add_reducer>(std::move(stream), algb::identity_mapper{data.data()},
                                        data.size());
    xens::get_scheduler().run();
    basdv::synchronize_device();
    REQUIRE(res.value() == data[0] + data[1]);
  }

  SECTION("we can reduce small data sets") {
    for (unsigned n = 3; n < 67; ++n) {
      auto data = make_dataset(rng, n);
      auto res = reduce<test_add_reducer>(basdv::stream{}, algb::identity_mapper{data.data()},
                                          data.size());
      xens::get_scheduler().run();
      basdv::synchronize_device();
      REQUIRE(res.value() == compute_sum(data));
    }
  }

  SECTION("we can reduce datasets that are a power of two") {
    for (unsigned int n = 2; n < (1u << 12u); n <<= 1u) {
      auto data = make_dataset(rng, n);
      auto res = reduce<test_add_reducer>(basdv::stream{}, algb::identity_mapper{data.data()},
                                          data.size());
      xens::get_scheduler().run();
      basdv::synchronize_device();
      REQUIRE(res.value() == compute_sum(data));
    }
  }

  SECTION("we can reduce arbitrary random data sets") {
    std::uniform_int_distribution<unsigned int> dist{128, 1u << 15u};
    for (int i = 0; i < 10; ++i) {
      auto n = dist(rng);
      auto data = make_dataset(rng, n);
      auto res = reduce<test_add_reducer>(basdv::stream{}, algb::identity_mapper{data.data()},
                                          data.size());
      xens::get_scheduler().run();
      basdv::synchronize_device();
      REQUIRE(res.value() == compute_sum(data));
    }
  }
}
